import torch
import torch.nn as nn
from model.AGCRNCell import AGCRNCell
import lib.utils as utils
# [Day 1 新增] 引入 NTK 工具
from model.NTK import compute_spectral_mask 

class AGCRN(nn.Module):
    def __init__(self, args):
        super(AGCRN, self).__init__()
        self.num_nodes = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.args = args

        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, args.embed_dim), requires_grad=True)
        if self.args.active_mode == "adptpolu":
            self.poly_coefficients = nn.Parameter(torch.randn(1, args.act_k+1), requires_grad=True)
        else: self.poly_coefficients = None

        # 这里调用了 AVWDCRNN，所以该类必须在文件中定义
        self.encoder = AVWDCRNN(self.args, self.num_nodes, args.input_dim, args.rnn_units,
                                args.embed_dim, args.num_layers)

        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        
        # [Day 1 新增] 用于存储各层输入的字典
        self.layer_inputs = {}
        self.hook_handles = []
        self._register_hooks()

    # [Day 1 新增] 注册 Hook
    def _register_hooks(self):
        def get_input_hook(name):
            def hook(model, input, output):
                # input 是一个 tuple，取第一个元素
                if model.training: # 只在训练模式下记录，节省显存
                    self.layer_inputs[name] = input[0].detach() 
            return hook

        # 为 end_conv 注册 hook
        self.hook_handles.append(self.end_conv.register_forward_hook(get_input_hook("end_conv")))

    def forward(self, source, node_indices=None):
        # 注意：这里参数接口可能需要根据你 BasicTrainer 的调用调整
        # source: B, T_1, N, D
        
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings, self.poly_coefficients)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_nodes)
        output = output.permute(0, 1, 3, 2)                          #B, T, N, C

        return output

    def comm_socket(self, msg, device=None):
        self.args.socket.send(msg)
        if device:
            return self.args.socket.recv().to(device)
        else:
            return self.args.socket.recv()
    
    # [Day 1 核心修改] 实现 DSTI 稀疏上传逻辑
    def fedavg(self):
        # 1. 处理 Embedding
        if self.args.active_mode == "adptpolu":
            mean_p = self.comm_socket(self.poly_coefficients.data, self.args.device) / self.args.num_clients
            self.poly_coefficients = nn.Parameter(mean_p, requires_grad=True).to(mean_p.device)
        
        # 2. 处理 end_conv (应用 NTK 稀疏化)
        target_layer = self.end_conv
        layer_name = "end_conv"
        
        # 检查是否有 Input 和 Grad 信息可用
        has_grad = target_layer.weight.grad is not None
        has_input = layer_name in self.layer_inputs
        
        if has_grad and has_input:
            input_data = self.layer_inputs[layer_name]
            
            # 调用 NTK 工具计算 Mask
            # 默认压缩率 1% (0.01)
            compression_rate = getattr(self.args, 'compression_rate', 0.01)
            
            mask = compute_spectral_mask(
                model_layer=target_layer,
                param_name='weight',
                input_data=input_data, 
                layer_grad=target_layer.weight.grad,
                compression_rate=compression_rate
            )
            
            # 构建稀疏更新包
            sparse_indices = torch.nonzero(mask.view(-1)).squeeze()
            sparse_values = target_layer.weight.data.view(-1)[sparse_indices]
            
            msg = {
                'end_conv.weight': {
                    'type': 'sparse_update',
                    'indices': sparse_indices.cpu(), 
                    'values': sparse_values.cpu(),
                    'shape': target_layer.weight.shape
                },
                'end_conv.bias': target_layer.bias.data.cpu() 
            }
            # print(f">> DSTI Active: Sending {len(sparse_indices)} params for {layer_name}")
            updated_state_dict = self.comm_socket(msg)
            self.end_conv.load_state_dict(updated_state_dict)
        else:
            # Fallback: 全量更新
            # print(">> DSTI Fallback: Full update (No gradients found)")
            model_dict = self.comm_socket(self.end_conv.state_dict())
            self.end_conv.load_state_dict(model_dict)
            
        # 3. 处理 Encoder (递归调用)
        self.encoder.fedavg()

# [关键修复] 必须保留 AVWDCRNN 类的定义
class AVWDCRNN(nn.Module):
    def __init__(self, args, node_num, dim_in, dim_out, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.args = args
        self.dcrnn_cells.append(AGCRNCell(self.args, node_num, dim_in, dim_out, embed_dim))

        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(self.args, node_num, dim_out, dim_out, embed_dim))

    def forward(self, x, init_state, node_embeddings, poly_coefficients):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim, (x.shape, self.node_num)
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings, poly_coefficients)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states
    
    def fedavg(self):
        for model in self.dcrnn_cells: model.fedavg()