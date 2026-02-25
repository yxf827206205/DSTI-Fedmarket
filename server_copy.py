from lib.server_socket import ServerSocket
import argparse
import time
import copy
import torch
import collections


def reconstruct_sparse_tensor(sparse_msg, base_tensor, device):
    """
    将稀疏消息还原为稠密张量。
    
    Args:
        sparse_msg: 客户端发送的字典 {'type': 'sparse', 'indices': ..., 'values': ...}
        base_tensor: 上一轮的全局模型参数 (作为背景板)
        device: 目标设备
    """
    # 1. 创建上一轮权重的副本
    reconstructed = base_tensor.clone().to(device)
    
    # 2. 提取稀疏更新
    indices = sparse_msg['indices'].to(device)
    values = sparse_msg['values'].to(device)
    
    # 3. 将更新应用到副本上 (覆盖旧值 或 累加梯度，取决于你的DSTI实现)
    # DSTI通常是传输"变化后的重要参数"，所以这里我们做覆盖更新
    # 注意：indices 需要是 flat 的或者是坐标格式，这里假设是 flatten 后的索引
    # 如果是二维索引，使用 sparse_msg['indices'] 直接索引需要注意维度匹配
    
    # 假设 indices 是 flatten 后的索引 (一维)
    reconstructed.view(-1)[indices] = values.view(-1)
    
    return reconstructed

def FedAvg(w, node_indices_list=None, N=None, dsp=None, dsu=None, device=None):
    """
    执行联邦平均算法，并处理嵌入参数。

    Parameters:
    - w: 客户端的权重列表。
    - node_indices_list: 客户端的节点索引列表（用于嵌入参数）。
    - N, dsp, dsu: 嵌入矩阵相关参数，仅在需要处理嵌入时传入。

    Returns:
    - 全局参数更新后的字典。
    """
    target_device = torch.device("cuda:5")
    torch.cuda.empty_cache()
    # 检查是否包含嵌入参数
    first_client_weights = w[0]
    contains_embedding = 'B_sp' in first_client_weights and 'B_su' in first_client_weights
    
    if contains_embedding and node_indices_list:
        # 处理嵌入参数
        global_B_sp = torch.zeros(N, dsp).to(target_device)  # 固定图结构嵌入矩阵
        global_B_su = torch.zeros(N, dsu).to(target_device)  # 未知空间信息嵌入矩阵
        
        # 遍历每个客户端，将其嵌入复制到全局矩阵的相应行
        for client_idx, client_weights in enumerate(w):
            node_indices = node_indices_list[client_idx]
            client_weights = {k: v.to(target_device) for k, v in client_weights.items()}
            node_indices = node_indices.to(target_device)
            global_B_sp[node_indices] = client_weights['B_sp'][node_indices]
            global_B_su[node_indices] = client_weights['B_su'][node_indices]
        
        # 返回处理好的全局嵌入矩阵
        return {'B_sp': global_B_sp.to('cpu'), 'B_su': global_B_su.to('cpu')}
    
    # 若不包含嵌入参数，则执行标准的联邦平均
    

    w_avg = copy.deepcopy(w[0])
    # 将所有权重迁移到目标设备
    for k in w_avg.keys():
        w_avg[k] = w_avg[k].to(target_device)
    # 确保所有权重在统一的设备上
    for i in range(len(w)):
        for k in w[i].keys():
            w[i][k] = w[i][k].to(target_device)
    
    # 计算加权平均
    for k in w_avg.keys():
        # 跳过嵌入参数，以防重复处理
        if k not in ['B_sp', 'B_su']:
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
    for k in w_avg.keys():
        w_avg[k] = w_avg[k].to('cpu')
    return w_avg


class Server():
    def __init__(self, n_clients, port, ip, N, dsp, dsu, device):
        self.socket = ServerSocket(n_clients, port, ip)
        
        while True:
            rcvd_msgs = self.socket.recv()
            if rcvd_msgs:
                if isinstance(rcvd_msgs[0], dict) and 'node_indices' in rcvd_msgs[0]:
                    # 收到嵌入参数，提取每个客户端的 node_indices 和嵌入参数权重
                    node_indices_list = [msg['node_indices'] for msg in rcvd_msgs]
                    weights_list = [msg['weights'] for msg in rcvd_msgs]
                    
                    # 进行嵌入参数的行替换更新
                    self.socket.send(FedAvg(weights_list, node_indices_list, N, dsp, dsu, device))
                
                elif type(rcvd_msgs[0]) == collections.OrderedDict or type(rcvd_msgs[0]) == dict:
                    # 收到普通模型参数，直接进行联邦平均
                    self.socket.send(FedAvg(rcvd_msgs))
                
                
                
                else:
                    # 处理其他类型的消息（例如数值累加）
                    self.socket.send(sum(rcvd_msgs))
            
            else:
                print("[SERVER RECVED NONE]")
                self.socket.close()
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='n')
    parser.add_argument('-p', dest='port')
    parser.add_argument('-i', dest='ip')
    parser.add_argument('-N', dest='N', type=int, help="Total number of nodes")
    parser.add_argument('-dsp', dest='dsp', type=int, help="Dimension of known structure embedding")
    parser.add_argument('-dsu', dest='dsu', type=int, help="Dimension of unknown space embedding")
    parser.add_argument('--device', dest='device', default='cuda:0', help="Specify the device to use (e.g., 'cuda:0', 'cpu')")

    args = parser.parse_args()

    # 初始化服务器并传入必要参数
    server = Server(
        n_clients=int(args.n), 
        port=int(args.port), 
        ip=args.ip,
        N=args.N,
        dsp=args.dsp,
        dsu=args.dsu,
        device=args.device
    )