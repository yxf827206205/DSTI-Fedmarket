import copy, torch
import torch.nn as nn
import numpy as np
import pickle
def split(ary, indices_or_sections):
    import numpy.core.numeric as _nx
    Ntotal = len(ary)
    Nsections = int(indices_or_sections)
    Neach_section, extras = divmod(Ntotal, Nsections)
    section_sizes = ([0] +
                        extras * [Neach_section+1] +
                        (Nsections-extras) * [Neach_section])
    div_points = _nx.array(section_sizes, dtype=_nx.intp).cumsum()

    sub_arys = []
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_arys.append(ary[st:end])

    return sub_arys


def avg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

class SpatialEmbedding(nn.Module):
    def __init__(self, args, N, dsp, dsu):
        super(SpatialEmbedding, self).__init__()
        # 邻接矩阵 A
        # self.A = torch.tensor(A, dtype=torch.float32)  # 需要在训练之前设置
        self.args = args
        self.N = N
        # 可学习的代码本
        self.B_sp = nn.Parameter(torch.randn(N, dsp))  # 固定图结构嵌入
        self.B_su = nn.Parameter(torch.randn(N, dsu))  # 未知空间信息嵌入
        self.node_indices = None
    # def set_adjacency_matrix(self, A):
    #     self.A = torch.tensor(A, dtype=torch.float32)

    def forward(self, node_indices):
        self.node_indices = node_indices
        # 计算固定图结构的嵌入
        mask = torch.ones(self.N, dtype=bool)
        mask[node_indices] = False
        # with torch.no_grad():
        #     self.B_sp.data[mask] = 0
        #     self.B_su.data[mask] = 0
        # Esp = torch.matmul(self.A, self.B_sp)  # AB_sp = E_sp
        Esp = self.B_sp
        # 选择节点 i 的嵌入向量 E_i_sp
        E_i_sp = Esp[node_indices]  # 选择对应的行

        # 计算未知空间信息的嵌入
        # E_su = torch.matmul(self.A, self.B_su)  # AB_su = E_su
        E_i_su = self.B_su[node_indices]  # 选择对应的行

        return E_i_sp, E_i_su
    def comm_socket(self, msg, device=None):
        self.args.socket.send(msg)
        if device:
            return self.args.socket.recv().to(device)
        else:
            return self.args.socket.recv()
    def fedavg(self):
        # mean_p = self.comm_socket(self.B_sp, self.args.device) 
        # self.B_sp.data = mean_p.data / self.args.num_clients
        # mean_p = self.comm_socket(self.B_su, self.args.device) 
        # self.B_su.data = mean_p.data / self.args.num_clients
        
        # 上传本地嵌入参数
        msg = {
            'node_indices': self.node_indices,  # 客户端负责的节点索引
            'weights': {                        # 权重包含嵌入参数
                'B_sp': self.B_sp.data.clone(),
                'B_su': self.B_su.data.clone()
            }
        }
        # self.comm_socket(msg)  # 发送嵌入数据到服务器

        # 接收服务器发送的全局嵌入矩阵
        updated_embeddings = self.comm_socket(msg)

        # 更新本地嵌入矩阵
        if 'B_sp' in updated_embeddings and 'B_su' in updated_embeddings:
            self.B_sp.data = updated_embeddings['B_sp'].to(self.args.device)
            self.B_su.data = updated_embeddings['B_su'].to(self.args.device)



def smooth_fill_zeros(matrix):
    """
    对 [steps, nodes] 形状的矩阵按 steps 维度进行平滑插值填充 0 值
    :param matrix: 形状为 [steps, nodes] 的 numpy 数组
    :return: 填充后的 numpy 数组
    """
    steps, nodes = matrix.shape
    for node in range(nodes):
        # 获取当前节点的数据
        data = matrix[:, node]

        # 将 0 值视为需要填充的缺失值
        indices = np.arange(steps)
        non_zero_mask = data != 0
        
        # 如果整个节点列都是 0，跳过这个节点
        if not np.any(non_zero_mask):
            continue
        
        # 使用线性插值填充 0 值
        filled_data = np.interp(indices, indices[non_zero_mask], data[non_zero_mask])

        # 用填充好的数据替换原来的数据
        matrix[:, node] = filled_data

    return matrix