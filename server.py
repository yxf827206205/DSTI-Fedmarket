from lib.server_socket import ServerSocket
import argparse
import copy
import torch
import collections
import torch.nn.functional as F
import numpy as np
import json
from openai import OpenAI

# ================= [Helper: Robust Sparse Reconstruction] =================
def reconstruct_sparse_tensor(sparse_msg, base_tensor, device):
    """
    DSTI 核心重构逻辑：
    将稀疏的 updates (indices, values) 覆盖到 base_tensor (旧模型) 上。
    """
    # 必须 clone，否则会修改引用源（即 global_model_prev），导致历史污染
    reconstructed = base_tensor.to(device).clone()
    indices = sparse_msg['indices'].to(device)
    values = sparse_msg['values'].to(device)
    
    # 展平并赋值 (Scatter Update)
    reconstructed.view(-1)[indices] = values
    return reconstructed

# ================= [Renovated: Event-Aware Logic Agent] =================
class Data_Driven_LLM_Agent:
    def __init__(self):
        # 建议换成你的实际 Key，或者从环境变量读取
        self.client = OpenAI(
            api_key="sk-c612e9714f504649bc001e702ca968d8", 
            base_url="https://api.deepseek.com/v1",
            timeout=30.0 
        )
        print("[Agent] Initialized: Event-Aware Logic Mode.")

    def _extract_features(self, logits):
        """
        提取用于区分“噪声”和“事件”的物理指纹
        """
        if logits is None: return None
        
        # logits: (Batch, Time, Nodes, 1)
        data = logits.detach().cpu().numpy()
        
        # 1. 空间聚集度 (通过熵判断：低熵=聚焦/有结构，高熵=散乱/噪声)
        squeezed = logits.squeeze(-1) # (B, T, N)
        probs = F.softmax(squeezed, dim=-1).detach().cpu().numpy()
        # 计算每个时间步的熵
        raw_ent = -np.sum(probs * np.log(probs + 1e-9), axis=-1)
        # 归一化 (0~1)
        norm_ent = np.mean(raw_ent) / np.log(probs.shape[-1])
        
        # 2. 波动幅度 (标准差)
        std = np.std(data)
        
        # 3. 流量强度 (均值，车祸通常意味着流量激增或骤减)
        mean = np.mean(data)
        
        return {
            "std": float(std),
            "entropy": float(norm_ent),
            "mean": float(mean)
        }

    def analyze_batch(self, logits_map, device):
        """
        Batch 逻辑推理：不仅找不同，还要判断不同的原因 (鉴别诊断)
        logits_map: {client_idx: logits}
        """
        # 1. 构建群体状态报告
        batch_report = []
        valid_indices = []
        
        for idx, logits in logits_map.items():
            if logits is None: continue
            feat = self._extract_features(logits)
            valid_indices.append(idx)
            
            # 我们不再直接告诉 LLM 它是 "Pattern A"，而是给它原始特征让它分析
            batch_report.append(
                f"Client {idx}: Std={feat['std']:.3f} (Volatility), Entropy={feat['entropy']:.3f} (Randomness 0-1), Mean={feat['mean']:.3f} (Intensity)"
            )
            
        if not batch_report:
            return None, "No Data"

        report_str = "\n".join(batch_report)
        
        # 2. 构造“鉴别诊断” Prompt (核心逻辑)
        prompt = f"""
        You are a Traffic Event Discriminator for a Federated Learning system.
        You have data from {len(valid_indices)} sensors. Your goal is to distinguish between **True Traffic Events** (Accidents/Congestion) and **Sensor Failures** (Noise).
        
        Sensor Data Report:
        {report_str}
        
        Decision Logic (Physics-Based):
        1. **The Baseline**: Identify the majority behavior. This is the "Background Traffic".
        2. **The Outliers**: Identify clients that deviate significantly in 'Std' or 'Mean'.
        3. **The Discrimination (CRITICAL)**:
           - **Case A: Valid Event (High Value)** -> If an outlier has **High Std** BUT **Low Entropy (< 0.8)**. 
             *Reasoning*: Traffic events are structured (concentrated at specific nodes/intersections). 
             *Action*: Assign High Weight (1.0).
             
           - **Case B: Sensor Failure (Harmful)** -> If an outlier has **High Std** AND **High Entropy (> 0.9)**.
             *Reasoning*: Noise is random and unstructured across the whole map.
             *Action*: Assign Low Weight (0.1).
             
           - **Case C: Normal Traffic** -> Matches the baseline or has low volatility.
             *Action*: Assign Medium Weight (0.5).
        
        Task: Analyze each client and assign a confidence score (0.0 - 1.0).
        Output JSON format: {{ "0": 0.5, "1": 1.0, ... }}
        """

        try:
            # print(f"\n[Agent] Thinking about {len(valid_indices)} clients...")
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={ "type": "json_object" } # 强制 JSON
            )
            
            # 3. 解析结果
            result_json = json.loads(response.choices[0].message.content)
            
            # 映射回列表顺序
            scores = []
            for idx in sorted(logits_map.keys()):
                # JSON key 是 str，需要转 int 匹配
                s = result_json.get(str(idx), result_json.get(int(idx), 0.5)) 
                scores.append(s)
                
            return scores, result_json

        except Exception as e:
            print(f"[Agent Error] {e}. Fallback to statistical filtering.")
            # 兜底逻辑：基于方差的 Z-Score 剔除
            stds = []
            valid_ids = []
            for i in sorted(logits_map.keys()):
                 if logits_map[i] is not None:
                     stds.append(self._extract_features(logits_map[i])['std'])
                     valid_ids.append(i)
            
            if not stds: return [1.0]*len(logits_map), "Fallback"
            
            avg_std = np.mean(stds)
            scores_map = {}
            for idx in valid_ids:
                feat = self._extract_features(logits_map.get(idx))
                # 如果方差异常高 (3倍均值) 且 熵也高 (>0.9)，判为噪声
                if feat['std'] > 3 * avg_std and feat['entropy'] > 0.9:
                    scores_map[idx] = 0.1
                else:
                    scores_map[idx] = 1.0
            
            # 对齐返回列表
            final_scores = []
            for idx in sorted(logits_map.keys()):
                final_scores.append(scores_map.get(idx, 1.0))
                
            return final_scores, "Statistical Fallback"

# ================= [Renovated: FedAvg with DSTI & Logic] =================

def FedAvg(w, node_indices_list=None, N=None, dsp=None, dsu=None, device=None, global_model_prev=None, logits_list=None, llm_agent=None):
    target_device = torch.device(device if device else "cuda:0")
    
    # --- 1. Embedding Aggregation (处理 B_sp, B_su) ---
    if 'B_sp' in w[0] and node_indices_list:
        global_B_sp = torch.zeros(N, dsp).to(target_device)
        global_B_su = torch.zeros(N, dsu).to(target_device)
        # 记录每个节点被更新的次数，用于取平均
        update_counts = torch.zeros(N).to(target_device) 
        
        for client_idx, client_weights in enumerate(w):
            node_indices = node_indices_list[client_idx].to(target_device)
            c_sp = client_weights['B_sp'].to(target_device)
            c_su = client_weights['B_su'].to(target_device)
            
            global_B_sp[node_indices] += c_sp[node_indices]
            global_B_su[node_indices] += c_su[node_indices]
            update_counts[node_indices] += 1
            
        # 避免除以 0
        update_counts[update_counts == 0] = 1.0
        global_B_sp = global_B_sp / update_counts.unsqueeze(-1)
        global_B_su = global_B_su / update_counts.unsqueeze(-1)
        
        return {'B_sp': global_B_sp.to('cpu'), 'B_su': global_B_su.to('cpu')}

    # --- 2. Logic-Driven Weight Calculation ---
    norm_weights = None
    if logits_list and any(l is not None for l in logits_list) and llm_agent:
        # 构造 {idx: logits} map
        logits_map = {i: l for i, l in enumerate(logits_list)}
        
        print(f"\n>>> [Logic-Driven Aggregation]")
        raw_scores, details = llm_agent.analyze_batch(logits_map, target_device)
        
        if raw_scores:
            total_s = sum(raw_scores) + 1e-9
            norm_weights = [s / total_s for s in raw_scores]
            print(f"    LLM Scores: {details}")
            print(f"    Final Norm Weights: {[round(x, 3) for x in norm_weights]}")
        else:
            print("    [Info] No valid scores, using average.")

    # --- 3. Robust Parameter Aggregation (Dense + Sparse) ---
    processed_w = []
    valid_clients = [] # 记录有效客户端的索引
    
    for client_idx in range(len(w)):
        client_w_dense = {}
        # 跳过空包
        if not w[client_idx]: continue
        
        for k, v in w[client_idx].items():
            # 过滤元数据
            if k in ['type', 'shape', 'indices', 'values', '__layer_id__', 'logits']: continue
            
            # A. 处理稀疏更新 (DSTI)
            if isinstance(v, dict) and v.get('type') == 'sparse_update':
                # [关键修正] 必须使用 global_model_prev 作为底板
                if global_model_prev is not None and k in global_model_prev:
                    base = global_model_prev[k]
                else:
                    # 如果没有历史底板（例如Server刚重启或第一轮DSTI），使用全零兜底
                    # Client 端的熔断机制会保护模型不被破坏
                    # print(f"    [Warning] No base for {k}, using zeros.")
                    base = torch.zeros(v['shape']).to(target_device)
                
                client_w_dense[k] = reconstruct_sparse_tensor(v, base, target_device)
            
            # B. 处理全量更新 (Tensor)
            elif isinstance(v, torch.Tensor):
                client_w_dense[k] = v.to(target_device)
        
        processed_w.append(client_w_dense)
        valid_clients.append(client_idx)

    # --- 4. Weighted Averaging ---
    if not processed_w: return {}
    
    w_avg = {}
    ref_keys = processed_w[0].keys()
    
    # 即使某个 key 只有部分 client 有，也应该处理 (Robustness)
    for k in ref_keys:
        # 初始化聚合容器
        w_acc = torch.zeros_like(processed_w[0][k])
        total_weight = 0.0
        
        for i, client_idx in enumerate(valid_clients):
            # 获取该 Client 的权重 (如果 LLM 没算出来，就平均)
            weight = norm_weights[client_idx] if norm_weights else (1.0 / len(valid_clients))
            
            if k in processed_w[i]:
                w_acc += processed_w[i][k] * weight
                total_weight += weight
        
        # 归一化 (防止权重和不为1)
        if total_weight > 0:
            w_avg[k] = w_acc / total_weight
        else:
            w_avg[k] = w_acc # Should not happen

    return {k: v.to('cpu') for k, v in w_avg.items() if isinstance(v, torch.Tensor)}

# ================= [Server Main Loop] =================
class Server():
    def __init__(self, n_clients, port, ip, N, dsp, dsu, device):
        self.socket = ServerSocket(n_clients, port, ip)
        self.device = device
        # 存储每层的全局状态 {layer_id: state_dict}
        self.layer_models = {} 
        self.llm_agent = Data_Driven_LLM_Agent()
        print(f"Server initialized on {device}. Waiting for {n_clients} clients...")

        while True:
            try:
                rcvd_msgs = self.socket.recv()
            except Exception as e:
                print(f"[Server Error] {e}")
                break
            
            if rcvd_msgs:
                # Case 1: Embedding Update (Special Handling)
                if isinstance(rcvd_msgs[0], dict) and 'node_indices' in rcvd_msgs[0]:
                    node_indices_list = [msg['node_indices'] for msg in rcvd_msgs]
                    weights_list = [msg['weights'] for msg in rcvd_msgs]
                    result = FedAvg(weights_list, node_indices_list, N, dsp, dsu, self.device)
                    self.socket.send(result)
                
                # Case 2: Layer-wise Update (DSTI / Standard)
                elif isinstance(rcvd_msgs[0], (dict, collections.OrderedDict)):
                    # 获取这一批更新是针对哪一层的
                    # 注意：假设所有 Client 这一轮同步更新同一层 (TCP 顺序保证)
                    layer_id = rcvd_msgs[0].get('__layer_id__')
                    logits_list = [msg.get('logits') for msg in rcvd_msgs]
                    
                    # 获取该层上一轮的全局模型 (作为 DSTI Base)
                    prev_model = self.layer_models.get(layer_id) if layer_id else None
                    
                    new_weights = FedAvg(
                        w=rcvd_msgs, 
                        device=self.device,
                        global_model_prev=prev_model,
                        logits_list=logits_list,
                        llm_agent=self.llm_agent,
                        N=N, dsp=dsp, dsu=dsu
                    )
                    
                    # 更新全局状态
                    if layer_id:
                        self.layer_models[layer_id] = copy.deepcopy(new_weights)
                        
                    self.socket.send(new_weights)
                
                # Case 3: Scalar/Other (Fallback)
                else:
                    self.socket.send(sum(rcvd_msgs))
            else:
                print("[SERVER RECVED NONE] Shutting down.")
                self.socket.close()
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='n')
    parser.add_argument('-p', dest='port')
    parser.add_argument('-i', dest='ip')
    parser.add_argument('-N', dest='N', type=int, default=228)
    parser.add_argument('-dsp', dest='dsp', type=int, default=32)
    parser.add_argument('-dsu', dest='dsu', type=int, default=32)
    parser.add_argument('--device', dest='device', default='cuda:0')

    args = parser.parse_args()
    server = Server(int(args.n), int(args.port), args.ip, args.N, args.dsp, args.dsu, args.device)