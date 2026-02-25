import torch.nn as nn
import torch
import pickle
from lib.utils import SpatialEmbedding
from .layer import *
from torch.func import functional_call, vmap, grad, vjp
import torch.nn.functional as F
from model.NTK import compute_spectral_mask 

# ================= [Modified BasicBlock with DSTI & Identity] =================
class BasicBlock(nn.Module):
    def __init__(self, args, dX, dropout=0.1, name="block"):
        super(BasicBlock, self).__init__()
        self.linear = nn.Linear(dX, dX)
        self.norm = nn.LayerNorm(dX)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.args = args
        self.layer_id = name  # [Fix] 身份证号
        
        # [DSTI] 缓存输入用于计算 NTK
        self.layer_input = None
        self._register_hooks()

    def _register_hooks(self):
        def hook(model, input, output):
            if model.training:
                self.layer_input = input[0].detach()
        self.linear.register_forward_hook(hook)

    def forward(self, x):
        residual = x
        out = self.linear(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = out + residual
        return out
        
    def comm_socket(self, msg, device=None):
        self.args.socket.send(msg)
        if device:
            return self.args.socket.recv().to(device)
        else:
            return self.args.socket.recv()

    def fedavg(self):
        # 1. 检查是否满足稀疏化条件
        has_grad = (self.linear.weight.grad is not None)
        has_input = (self.layer_input is not None)
        compression_rate = getattr(self.args, 'compression_rate', 0.01)
        
        if has_grad and has_input:
            try:
                # A. 计算谱掩码
                mask = compute_spectral_mask(
                    model_layer=self.linear,
                    param_name='weight',
                    input_data=self.layer_input,
                    layer_grad=self.linear.weight.grad,
                    compression_rate=compression_rate
                )
                
                # B. 构建稀疏包
                sparse_indices = torch.nonzero(mask.view(-1)).squeeze()
                sparse_values = self.linear.weight.data.view(-1)[sparse_indices]
                
                weight_msg = {
                    'type': 'sparse_update',
                    'indices': sparse_indices.cpu(),
                    'values': sparse_values.cpu(),
                    'shape': self.linear.weight.shape
                }
                
                # C. [Fix] 必须带上 __layer_id__，否则服务器会把底板清零！
                msg = {
                    'weight': weight_msg,
                    'bias': self.linear.bias.data.cpu(),
                    '__layer_id__': self.layer_id 
                }
                
                # [Log] 打印压缩信息 (取消注释以查看效果)
                print(f">> [{self.layer_id}] DSTI Sending {len(sparse_indices)} params ({compression_rate*100}%)")
                
                updated_state = self.comm_socket(msg)
                
                if isinstance(updated_state, dict) and 'weight' in updated_state:
                     self.linear.load_state_dict(updated_state)
                
            except Exception as e:
                print(f"[DSTI Error in {self.layer_id}] Fallback: {e}")
                # Fallback 时也要带 ID
                state = self.linear.state_dict()
                state['__layer_id__'] = self.layer_id
                model_dict = self.comm_socket(state)
                self.linear.load_state_dict(model_dict)
        else:
            # D. 全量更新 (First Round)
            state = self.linear.state_dict()
            state['__layer_id__'] = self.layer_id
            model_dict = self.comm_socket(state)
            self.linear.load_state_dict(model_dict)

        # 3. Norm 层 (参数少，全量，但也最好带个 ID 以防万一，虽然 server 代码对 Tensor 是直接 Avg)
        # 这里的 Norm 发送的是 OrderedDict，server 会识别为 layer_id=None 的普通聚合，
        # 因为 Norm 是标准 Avg，不需要历史状态，所以不带 ID 也没事。
        model_dict = self.comm_socket(self.norm.state_dict())
        self.norm.load_state_dict(model_dict)

# [Fix] MLPModule 也要支持命名
class MLPModule(nn.Module):
    def __init__(self, args, dX, num_blocks=1, dropout=0.1, name="MLP"):
        super(MLPModule, self).__init__()
        # 给每个 Block 分配唯一名字：MLPA_blk0, MLPA_blk1 ...
        self.blocks = nn.ModuleList([
            BasicBlock(args, dX, dropout, name=f"{name}_blk{i}") 
            for i in range(num_blocks)
        ])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
        
    def fedavg(self):
        for block in self.blocks: block.fedavg()

# ----------------- 辅助类 (gtnet 等保持不变) -----------------
class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)
        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1
        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)
                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=residual_channels, kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels, kernel_size=(1, self.receptive_field-rf_size_j+1)))
                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                new_dilation *= dilation_exponential
        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1,1), bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)
        self.idx = torch.arange(self.num_nodes).to(device)

    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'
        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))
        if self.gcn_true:
            if self.buildA_true:
                if idx is None: adp = self.gc(self.idx)
                else: adp = self.gc(idx)
            else: adp = self.predefined_A
        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            if idx is None: x = self.norm[i](x,self.idx)
            else: x = self.norm[i](x,idx)
        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

# ----------------- STCIMwithGCN -----------------
class STCIMwithGCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_dim = args.input_dim
        self.steps_per_day= args.steps_per_day
        self.tod_embedding_dim = args.tod_embedding_dim
        self.dow_embedding_dim = args.dow_embedding_dim
        self.num_nodes = args.num_nodes
        self.num_client_nodes = args.num_client_nodes
        self.dsp = args.dsp
        self.dsu = args.dsu
        self.in_steps = args.in_steps
        self.out_steps = args.out_steps
        self.args = args
        self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        self.SpatialEmbedding = SpatialEmbedding(args, self.num_nodes, self.dsp, self.dsu)
        self.data_embedding = nn.Linear(self.input_dim, 64)
        
        # [Fix] 实例化时传入 name
        self.MLPA = MLPModule(args, self.tod_embedding_dim + self.dow_embedding_dim, 1, dropout=0.1, name="MLPA")
        self.MLPB = MLPModule(args, self.dsp + self.dsu, 1, dropout=0.1, name="MLPB")
        self.MLPC = MLPModule(args, 64, 1, dropout=0.1, name="MLPC")
        self.MLPD = MLPModule(args,  self.tod_embedding_dim + self.dow_embedding_dim+self.dsp + self.dsu, 1, dropout=0.1, name="MLPD")
        self.MLPE = MLPModule(args,  self.tod_embedding_dim + self.dow_embedding_dim+self.dsp + self.dsu + 64, 3, dropout=0.1, name="MLPE")
        self.MLPF = MLPModule(args, self.in_steps, 2, dropout=0, name="MLPF")
        
        self.steps_linear = nn.Linear(self.in_steps, self.out_steps)
        self.out_linear = nn.Linear(self.tod_embedding_dim + self.dow_embedding_dim+self.dsp + self.dsu + 64, 1)

        self.data_l = nn.Parameter(torch.randn(64))
        self.tod_embedding_l = nn.Parameter(torch.randn(self.tod_embedding_dim))
        self.dow_embedding_l = nn.Parameter(torch.randn(self.dow_embedding_dim))

        self.gcn = gtnet(gcn_true=True, buildA_true=True, gcn_depth=2, num_nodes=self.num_client_nodes, device=args.device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=1, node_dim=10, dilation_exponential=1, conv_channels=16, residual_channels=16, skip_channels=32, end_channels=64, seq_length=self.in_steps, in_dim=1, out_dim=self.out_steps, layers=2, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)
        self.MLPGCN = MLPModule(args, self.out_steps, 2, dropout=0.1, name="MLPGCN")
        self.out_linear_input = None
        self._register_out_hooks()
        
        # [DSTI Fix] 标记 DSTI 是否已初始化
        self.dsti_ready = False
        
    def _register_out_hooks(self):
        def hook(model, input, output):
            if model.training:
                self.out_linear_input = input[0].detach()
        self.out_linear.register_forward_hook(hook)

    def forward(self, x, node_indices):
        batchsize = x.shape[0]
        tod = x[..., 1].clone()
        dow = x[..., 2].clone()
        x = x[..., : self.input_dim]
        
        x_gcn = x.permute(0, 3, 2, 1)
        x_gcn = self.gcn(x_gcn)
        x_gcn = x_gcn.permute(0, 3, 2, 1)
        x_gcn = self.MLPGCN(x_gcn)
        x_gcn = x_gcn.permute(0, 3, 2, 1)
        x_gcn = x_gcn.squeeze(-1).permute(0, 2, 1).unsqueeze(-1)
        x_gcn = x_gcn.permute(0, 2, 1, 3)
        
        tod_emb = self.tod_embedding((tod * self.steps_per_day).long())
        dow_emb = self.dow_embedding(dow.long())

        spa_n_emb, spa_u_emb= self.SpatialEmbedding(node_indices)
        tem_emb = torch.cat((tod_emb, dow_emb), dim=-1)
        tem_emb = self.MLPA(tem_emb)
        spa_emb = torch.cat((spa_n_emb, spa_u_emb), dim=-1)

        spa_emb = spa_emb.unsqueeze(1)  
        spa_emb = spa_emb.repeat(1, self.in_steps, 1, 1)  
        spa_emb = self.MLPB(spa_emb)
        st_emb = torch.cat((spa_emb, tem_emb), dim=-1)
        st_emb = self.MLPD(st_emb)
        x = self.data_embedding(x)
        x = self.MLPC(x)

        step = torch.cat((st_emb, x), dim=-1)
        step = self.MLPE(step)
        step = step.permute(0, 3, 2, 1)
        step = self.MLPF(step)
        step = self.steps_linear(step)
        step = step.permute(0, 3, 2, 1)
        out = self.out_linear(step)
        out = out + x_gcn
        return out
        
    def comm_socket(self, msg, device=None):
        self.args.socket.send(msg)
        if device:
            return self.args.socket.recv().to(device)
        else:
            return self.args.socket.recv()
            
    def fedavg(self):
        # 1. 常规层 (Full update)
        model_dict = self.comm_socket(self.tod_embedding.state_dict())
        self.tod_embedding.load_state_dict(model_dict)
        model_dict = self.comm_socket(self.dow_embedding.state_dict())
        self.dow_embedding.load_state_dict(model_dict)
        model_dict = self.comm_socket(self.data_embedding.state_dict())
        self.data_embedding.load_state_dict(model_dict)
        model_dict = self.comm_socket(self.steps_linear.state_dict())
        self.steps_linear.load_state_dict(model_dict)
        
        logits_to_send = None
        if hasattr(self, 'current_logits'):
            logits_to_send = self.current_logits
            del self.current_logits

        # 2. DSTI 逻辑 (out_linear)
        # 同样需要加上 __layer_id__
        do_dsti = self.dsti_ready and (self.out_linear.weight.grad is not None) and (self.out_linear_input is not None)
        
        if do_dsti:
            try:
                compression_rate = getattr(self.args, 'compression_rate', 0.01)
                mask = compute_spectral_mask(
                    model_layer=self.out_linear,
                    param_name='weight',
                    input_data=self.out_linear_input,
                    layer_grad=self.out_linear.weight.grad,
                    compression_rate=compression_rate
                )
                sparse_indices = torch.nonzero(mask.view(-1)).squeeze()
                sparse_values = self.out_linear.weight.data.view(-1)[sparse_indices]
                
                weight_msg = {
                    'type': 'sparse_update',
                    'indices': sparse_indices.cpu(),
                    'values': sparse_values.cpu(),
                    'shape': self.out_linear.weight.shape
                }
                msg = {
                    'weight': weight_msg,
                    'bias': self.out_linear.bias.data.cpu(),
                    '__layer_id__': 'out_linear', 
                    'logits': logits_to_send
                }
                
                updated_state_dict = self.comm_socket(msg)
                if isinstance(updated_state_dict, dict) and 'weight' in updated_state_dict:
                    self.out_linear.load_state_dict(updated_state_dict)
                
            except Exception as e:
                print(f"[DSTI Error OutLinear] Fallback: {e}")
                state_dict = self.out_linear.state_dict()
                state_dict['__layer_id__'] = 'out_linear'
                if logits_to_send is not None:
                 state_dict['logits'] = logits_to_send

                model_dict = self.comm_socket(state_dict)
                self.out_linear.load_state_dict(model_dict)
        else:
             # Full Update
             state_dict = self.out_linear.state_dict()
             state_dict['__layer_id__'] = 'out_linear'
             if logits_to_send is not None:
                 state_dict['logits'] = logits_to_send
             
             model_dict = self.comm_socket(state_dict)
             self.out_linear.load_state_dict(model_dict)
             
             self.dsti_ready = True
        
        # 3. 递归调用 (现在支持带名字了)
        self.SpatialEmbedding.fedavg()
        self.MLPA.fedavg()
        self.MLPB.fedavg()
        self.MLPC.fedavg()
        self.MLPD.fedavg()
        self.MLPE.fedavg()
        self.MLPF.fedavg()