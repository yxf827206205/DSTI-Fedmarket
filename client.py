import os
import sys
import argparse
import configparser
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
current_file_path = os.path.abspath(__file__)
file_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(file_dir)

from lib.client_socket import ClientSocket
from model.STCIM import STCIMwithGCN as Network
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed, print_model_parameters, get_memory_usage
from lib.dataloader import get_dataloader
from lib.logger import get_logger
from data.dividing import *

# ==========================================
# 鲁棒性增强：智能参数预处理
# ==========================================
def preprocess_args():
    """
    检测并处理混合了位置参数（旧脚本）和关键字参数的情况。
    如果是旧脚本启动 (run_dsti.sh)，sys.argv 会包含大量占位符 '0'。
    我们将它们转换为 argparse 能识别的标准格式。
    """
    # 原始脚本参数结构假设:
    # 0:script, 1:dataset, 2:mode, 3-11:placeholder(0), 12:device, 13+:flags
    
    # 简单的启发式检查：如果参数足够多，且第3到第5个参数都是'0'，则认为是旧脚本模式
    is_legacy_script_mode = (len(sys.argv) > 12) and \
                            (sys.argv[3] == '0') and \
                            (sys.argv[4] == '0')

    if is_legacy_script_mode:
        print("[Info] Detected legacy script arguments. Converting to standard flags...")
        
        # 提取关键参数
        dataset = sys.argv[1]
        mode = sys.argv[2]
        device = sys.argv[12]
        
        # 提取剩余的 flag 参数 (从 index 13 开始)
        remaining_flags = sys.argv[13:]
        
        # 重构 sys.argv，清除占位符，转换为标准格式
        # 注意：sys.argv[0] 是脚本名，必须保留
        new_argv = [sys.argv[0]]
        new_argv.extend(['--dataset', dataset])
        new_argv.extend(['--exp_mode', mode])
        new_argv.extend(['--device', device])
        new_argv.extend(remaining_flags)
        
        sys.argv = new_argv

def main(args):
    # init seed again inside main to be safe
    init_seed(args.seed)
    
    # load dataset
    train_loader, val_loader, test_loader, scaler = get_dataloader(args,
                                                                normalizer=args.normalizer,
                                                                single=False)
    # init model
    model = Network(args)
    model = model.to(args.device)
    
    # Model Initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.kaiming_normal_(p, mode='fan_in', nonlinearity='relu')
        else:
            nn.init.uniform_(p)

    args.logger.info(f"memory_usage: {get_memory_usage('cuda')}")

    # init loss function
    if args.loss_func == 'mask_mae':
        from lib.metrics import MAE_torch
        def masked_mae_loss(scaler, mask_value):
            def loss(preds, labels):
                if scaler:
                    preds = scaler.inverse_transform(preds)
                    labels = scaler.inverse_transform(labels)
                mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
                return mae
            return loss
        loss = masked_mae_loss(scaler, mask_value=0.0)
    elif args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(args.device)
    elif args.loss_func == 'huber':
        loss = torch.nn.HuberLoss(delta=1.0).to(args.device)
    elif args.loss_func == 'smae':
        loss = torch.nn.SmoothL1Loss().to(args.device)
    else:
        raise ValueError(f"Unknown loss function: {args.loss_func}")

    # init optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                                weight_decay=0.0001, amsgrad=False)
    
    # learning rate decay
    lr_scheduler = None
    if args.lr_decay:
        args.logger.info('Applying learning rate decay.')
        # lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
        #                                                     milestones=lr_decay_steps,
        #                                                     gamma=args.lr_decay_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, 
            T_max=args.epochs, 
            eta_min=1e-5
        )
    # start training
    trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                    args, lr_scheduler=lr_scheduler, logger=args.logger)

    print_model_parameters(model, trainer.logger, only_num=False)
    
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        # Robust checkpoint loading
        # [修复] 保持和 Trainer 中一致的相对路径
        ckpt_path = f'./checkpoints/{args.dataset}_{args.in_steps}_{args.out_steps}_{args.cid}.pth'
        
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path))
            args.logger.info(f"Load saved model from {ckpt_path}")
            trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
        else:
            args.logger.error(f"Checkpoint not found at {ckpt_path}")
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    try:
        # 1. 预处理参数（兼容旧脚本）
        preprocess_args()

        # 2. 定义参数解析器
        # 注意：这里我们不再从 sys.argv 硬读取 DATASET 和 MODE，而是让 argparse 处理
        # 为了兼容性，我们先解析出 DATASET 以便读取配置文件
        pre_parser = argparse.ArgumentParser(add_help=False)
        pre_parser.add_argument('--dataset', type=str, required=True)
        pre_parser.add_argument('--exp_mode', type=str, default='FED') # Default fallback
        
        # 只解析已知的 args，忽略其他的
        pre_args, _ = pre_parser.parse_known_args()
        
        DATASET = pre_args.dataset
        MODE = pre_args.exp_mode

        # 3. 读取配置文件
        config_file = f'./config/{DATASET}.conf'
        if not os.path.exists(config_file):
            # 尝试绝对路径
            config_file = os.path.join(file_dir, 'config', f'{DATASET}.conf')
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Configuration file not found: ./config/{DATASET}.conf")

        print(f'Read configuration file: {config_file}')
        config = configparser.ConfigParser()
        config.read(config_file)

        # 4. 定义完整参数
        parser = argparse.ArgumentParser(description='arguments')

        # Necessary args from command line (need to redefine here to be in the final args namespace)
        parser.add_argument('--dataset', default=DATASET, type=str)
        parser.add_argument('--exp_mode', default=MODE, choices=['CTR', 'SGL', 'FED'])
        parser.add_argument('--device', default='cuda:0', type=str, help='indices of GPUs')
        
        # CTR SGL FED config
        parser.add_argument('--num_clients', default=8, type=int)
        parser.add_argument('--cid', default=8, type=int)
        parser.add_argument('--divide', default="metis", type=str)

        # FED config
        parser.add_argument('--fedavg', default=False, action='store_true')
        parser.add_argument('--active_mode', default='softmax', choices=['softmax', 'sprtrelu', 'adptpolu'])
        parser.add_argument('--act_k', default=2, type=int)
        parser.add_argument('--local_epochs', default=2, type=int)
        parser.add_argument('-sp', '--server_port', dest='server_port', type=int)
        parser.add_argument('-sip', '--server_ip', dest='server_ip', type=str)
        parser.add_argument('-cp', '--self_port', dest='self_port', type=int)

        # Data config from file
        parser.add_argument('--steps_per_day', default=config['data'].getint('steps_per_day'), type=int)
        parser.add_argument('--tod_embedding_dim', default=config['data'].getint('tod_embedding_dim'), type=int)
        parser.add_argument('--dow_embedding_dim', default=config['data'].getint('dow_embedding_dim'), type=int)
        parser.add_argument('--num_nodes', default=config['data'].getint('num_nodes'), type=int)
        parser.add_argument('--num_client_nodes', default=0, type=int)
        parser.add_argument('--dsp', default=config['data'].getint('dsp'), type=int)
        parser.add_argument('--dsu', default=config['data'].getint('dsu'), type=int)
        parser.add_argument('--in_steps', default=config['data'].getint('in_steps'), type=int)
        parser.add_argument('--out_steps', default=config['data'].getint('out_steps'), type=int)
        parser.add_argument('--lag', default=config['data'].getint('lag'), type=int)
        parser.add_argument('--horizon', default=config['data'].getint('horizon'), type=int)
        parser.add_argument('--val_ratio', default=config['data'].getfloat('val_ratio'), type=float)
        parser.add_argument('--test_ratio', default=config['data'].getfloat('test_ratio'), type=float)
        parser.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
        parser.add_argument('--column_wise', default=eval(config['data']['column_wise']), type=eval)

        # Model config
        parser.add_argument('--embed_dim', default=config['model'].getint('embed_dim'), type=int)
        parser.add_argument('--rnn_units', default=config['model'].getint('rnn_units'), type=int)
        parser.add_argument('--num_layers', default=config['model'].getint('num_layers'), type=int)
        parser.add_argument('--accelerate', default=eval(config['model']['accelerate']), type=eval)
        parser.add_argument('--input_dim', default=config['model'].getint('input_dim'), type=int)
        parser.add_argument('--output_dim', default=config['model'].getint('output_dim'), type=int)
        parser.add_argument('--cheb_k', default=config['model'].getint('cheb_order'), type=int)

        # Train config
        parser.add_argument('--batch_size', default=config['train'].getint('batch_size'), type=int)
        parser.add_argument('--epochs', default=config['train'].getint('epochs'), type=int)
        parser.add_argument('--lr_init', default=config['train'].getfloat('lr_init'), type=float)
        parser.add_argument('--num_runs', default=config['train'].getint('num_runs'), type=int)
        parser.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
        parser.add_argument('--seed', default=config['train'].getint('seed'), type=int)
        parser.add_argument('--lr_decay', default=eval(config['train']['lr_decay']), type=eval)
        parser.add_argument('--lr_decay_rate', default=config['train'].getfloat('lr_decay_rate'), type=float)
        parser.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
        parser.add_argument('--early_stop', default=eval(config['train']['early_stop']), type=eval)
        parser.add_argument('--early_stop_patience', default=config['train'].getint('early_stop_patience'), type=int)
        parser.add_argument('--grad_norm', default=eval(config['train']['grad_norm']), type=eval)
        parser.add_argument('--max_grad_norm', default=config['train'].getint('max_grad_norm'), type=int)
        parser.add_argument('--real_value', default=eval(config['train']['real_value']), type=eval)

        # Test config
        parser.add_argument('--mae_thresh', default=eval(config['test']['mae_thresh']), type=eval)
        parser.add_argument('--mape_thresh', default=config['test'].getfloat('mape_thresh'), type=float)

        # Log config
        parser.add_argument('--log_dir', default='./', type=str)
        parser.add_argument('--log_step', default=config['log'].getint('log_step'), type=int)
        parser.add_argument('--plot', default=eval(config['log']['plot']), type=eval)

        # Static / Runtime config
        parser.add_argument('--mode', default='train', type=str)
        parser.add_argument('--debug', default=False, type=eval)
        parser.add_argument('--cuda', default=True, type=bool)
        
        # DSTI 参数
        parser.add_argument('--compression_rate', default=0.01, type=float, help='DSTI sparse compression rate')

        args = parser.parse_args()

        # Socket setup
        if args.server_port is None or args.self_port is None:
             # 如果是单机测试模式(SGL/CTR)可能不需要socket，但FED模式需要
             if args.exp_mode == 'FED':
                 raise ValueError("FED mode requires --server_port and --self_port")
             else:
                 args.socket = None
        else:
             args.socket = ClientSocket(int(args.cid), int(args.server_port), int(args.self_port), server_ip=args.server_ip)

        # Device Setup (Robust)
        init_seed(args.seed)
        if torch.cuda.is_available() and args.device != 'cpu':
            # 支持 "cuda:0", "cuda:1" 等格式
            try:
                device_id = 0
                if ':' in args.device:
                    device_id = int(args.device.split(':')[-1])
                torch.cuda.set_device(device_id)
                args.device = f'cuda:{device_id}'
            except Exception as e:
                print(f"[Warning] Failed to parse device ID from {args.device}, using cuda:0. Error: {e}")
                torch.cuda.set_device(0)
                args.device = 'cuda:0'
        else:
            args.device = 'cpu'

        print(f"Running on device: {args.device}")

        # Node config
        # 确保 dividing.py 里有对应的变量，或者使用更安全的方法获取
        try:
            # 动态获取 dividing.py 中的变量 (e.g., PeMSD7_8p_metis)
            var_name = f"{args.dataset}_{args.num_clients}p_{args.divide}"
            # 假设 dividing.py 的内容已经被导入到 globals() (from data.dividing import *)
            if var_name in globals():
                args.nodes_per = globals()[var_name]
            else:
                raise ValueError(f"Node division configuration '{var_name}' not found in data/dividing.py")
            
            args.nodes = args.nodes_per[args.cid-1]
            args.num_client_nodes = len(args.nodes)
        except Exception as e:
            print(f"[Error] Failed to load node division: {e}")
            raise

        # Log path setup
        current_dir = os.path.dirname(os.path.realpath(__file__))
        args.log_dir = os.path.join(current_dir, 'log', f"in{args.lag}_out{args.horizon}")
        args.logger = get_logger(args, args.log_dir, debug=args.debug)
        args.logger.info('Experiment log path in: {}'.format(args.log_dir))

        for _ in range(args.num_runs):
            main(args)

    except Exception as e:
        print(f"[CRITICAL ERROR] Program crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)