import torch
import math
import os
import time
import copy
import numpy as np
from lib.TrainInits import MAE_torch, RMSE_torch, MAPE_torch, All_Metrics


class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None, logger=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)

        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = logger
        
        # [修改点] 使用相对路径，并自动创建文件夹
        checkpoint_dir = './checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
            
        self.best_path = os.path.join(checkpoint_dir, f'{args.dataset}_{args.in_steps}_{args.out_steps}_{args.cid}.pth')

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        total_mae, total_rmse, total_mape = 0, 0, 0

        with torch.no_grad():
            for batch_idx, (data, target, val_node_indices) in enumerate(val_dataloader):
                torch.cuda.empty_cache()
                data = data.to(self.args.device)
                label = target[..., :self.args.output_dim].to(self.args.device)
                val_node_indices = val_node_indices.to(self.args.device)
                output = self.model(data, val_node_indices)
                # if self.args.real_value:
                #     label = self.scaler.inverse_transform(label)
                output = self.scaler.inverse_transform(output)
                # label = self.scaler.inverse_transform(label)
                loss = self.loss(output, label)
                total_val_loss += loss.item()

                total_mae += MAE_torch(output, label).item()
                total_rmse += RMSE_torch(output, label).item()
                total_mape += MAPE_torch(output, label).item()

        mae = total_mae / len(val_dataloader)
        rmse = total_rmse / len(val_dataloader)
        mape = total_mape / len(val_dataloader)
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: Average Loss: {:.6f}'.format(epoch, val_loss))
        self.logger.info('**********Val Epoch {}: MAE: {:.6f} RMSE: {:.6f} MAPE: {:.6f}'.format(epoch, mae, rmse, mape))
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_mae, total_rmse, total_mape = 0, 0, 0

        for batch_idx, (data, target, train_node_indices) in enumerate(self.train_loader):
            torch.cuda.empty_cache()
            data = data.to(self.args.device) # B, T_in, N, 1
            label = target[..., :self.args.output_dim].to(self.args.device)  # B, T_out, N, 1
            train_node_indices = train_node_indices.to(self.args.device)
            self.optimizer.zero_grad()

            #data and target shape: B, T, N, F; output shape: B, T, N, F
            output = self.model(data, train_node_indices)
            # if self.args.real_value:
            #     label = self.scaler.inverse_transform(label)
            # print(output.shape, label.shape)
            output = self.scaler.inverse_transform(output)
            # label = self.scaler.inverse_transform(label)
            loss = self.loss(output, label)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            total_mae += MAE_torch(output, label).item()
            total_rmse += RMSE_torch(output, label).item()
            total_mape += MAPE_torch(output, label).item()

            #log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))

        train_epoch_loss = total_loss/self.train_per_epoch
        mae = total_mae / self.train_per_epoch
        rmse = total_rmse / self.train_per_epoch
        mape = total_mape / self.train_per_epoch
        self.logger.info('**********Train Epoch {}: Average Loss: {:.6f}'.format(epoch, train_epoch_loss))
        self.logger.info('**********Train Epoch {}: MAE: {:.6f} RMSE: {:.6f} MAPE: {:.6f}'.format(epoch, mae, rmse, mape))

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            for ep in range(1, (self.args.local_epochs + 1) if self.args.fedavg else 2):
                torch.cuda.empty_cache()
                #epoch_time = time.time()
                train_epoch_loss = self.train_epoch(epoch)
                #print(time.time()-epoch_time)
                #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            #if self.val_loader == None:
            #val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
                # [修改点] 立即保存到硬盘，防止程序中断丢失模型
                if not self.args.debug:
                    torch.save(best_model, self.best_path)
        
            if self.args.fedavg:
                # 1. 先切换到评估模式，生成一份“未来预测信心”(Logits)
                # 这一步必须在 DSTI 计算之前做，避免梯度干扰
                self.model.eval()
                with torch.no_grad():
                    try:
                        # 获取一个验证集 batch 作为代表性样本
                        # 注意：这里我们重新创建一个迭代器以获取随机样本
                        val_batch = next(iter(self.val_loader))
                    except StopIteration:
                        self.val_loader_iter = iter(self.val_loader)
                        val_batch = next(self.val_loader_iter)
                    
                    v_data, v_target, v_indices = val_batch
                    v_data = v_data.to(self.args.device)
                    v_indices = v_indices.to(self.args.device)
                    
                    # 前向推理获取 Logits (B, T_out, N, 1)
                    logits = self.model(v_data, v_indices)
                    
                    # 将 Logits 暂存到 model 实例中
                    # fedavg() 会在发送消息时检查并带上这个 current_logits
                    # 我们只取第 0 步预测 (即未来第一个时间点) 以节省通信量，或者根据需求取全部
                    self.model.current_logits = logits.detach().cpu()
                # ================= [Day 2 结束] ===============
                # ================= [DSTI 核心逻辑: 必须切回 Train 模式] =================
                # 关键点：Hook 只有在 model.training=True 时才会记录 input
                # 所以这里必须强行切回 train()，否则 DSTI 稀疏化会失效！
                self.model.train() 
                
                # 2. 获取一个新的 Batch 数据用于计算梯度
                try:
                    data_batch = next(iter(self.train_loader))
                except StopIteration:
                    self.train_loader_iter = iter(self.train_loader)
                    data_batch = next(self.train_loader_iter)
                
                data, target, train_node_indices = data_batch
                
                # 3. 数据迁移
                data = data.to(self.args.device) 
                label = target[..., :self.args.output_dim].to(self.args.device)
                train_node_indices = train_node_indices.to(self.args.device)
                
                # 4. 前向传播 (Forward) -> 触发 AGCRN/STCIM 的 Hook 记录 Input
                output = self.model(data, train_node_indices)
                output = self.scaler.inverse_transform(output)
                
                # 5. 反向传播 (Backward) -> 生成 .grad 属性
                loss = self.loss(output, label)
                self.optimizer.zero_grad()
                loss.backward() 
                self.model.fedavg()
                self.optimizer.zero_grad()

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        #save the best model to file
        # if not self.args.debug:
        #     torch.save(best_model, self.best_path)
        #     self.logger.info("Saving current best model to " + self.best_path)

        #test
        self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target, test_node_indices) in enumerate(data_loader):
                torch.cuda.empty_cache()
                data = data.to(args.device)
                label = target[..., :args.output_dim].to(args.device)
                test_node_indices = test_node_indices.to(args.device)
                # output = model(data, target, teacher_forcing_ratio=0)
                output = model(data, test_node_indices)
                y_true.append(label)
                y_pred.append(output)
        # y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_true = torch.cat(y_true, dim=0)
        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        # np.save('./{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
        # np.save('./{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.4f}, RMSE: {:4f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape*100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
                    mae, rmse, mape*100))

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
    
    def val_epoch_save(self, val_dataloader):
        self.model.eval()

        MAE_per_nodes = [0 for _ in range(self.args.num_nodes)]
        RMSE_per_nodes = [0 for _ in range(self.args.num_nodes)]
        MAPE_per_nodes = [0 for _ in range(self.args.num_nodes)]

        with torch.no_grad():
            for batch_idx, (data, target, val_node_incides) in enumerate(val_dataloader):
                torch.cuda.empty_cache()
                data = data.to(self.args.device)
                label = target[..., :self.args.output_dim].to(self.args.device)
                val_node_incides = val_node_incides.to(self.args.device)
                output = self.model(data, val_node_incides)

                output = self.scaler.inverse_transform(output)
                # label = self.scaler.inverse_transform(label)

                for i in range(self.args.num_nodes):
                    output_i, label_i = output[:,:,i,:], label[:,:,i,:]
                    MAE_per_nodes[i] += MAE_torch(output_i, label_i).item()
                    RMSE_per_nodes[i] += RMSE_torch(output_i, label_i).item()
                    MAPE_per_nodes[i] += MAPE_torch(output_i, label_i).item()


        for i in range(self.args.num_nodes):
            MAE_per_nodes[i] /= len(val_dataloader)
            RMSE_per_nodes[i] /= len(val_dataloader)
            MAPE_per_nodes[i] /= len(val_dataloader)
        torch.save(torch.Tensor([MAE_per_nodes, RMSE_per_nodes, MAPE_per_nodes]), f'Error_{self.args.inter_dropout}.pth')