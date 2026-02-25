import torch
from torch.func import functional_call, vmap, grad, vjp
import torch.nn.functional as F
def compute_spectral_mask(model_layer, param_name, input_data, layer_grad, compression_rate=0.01):
    """
    计算谱重要性掩码 (Spectral Importance Mask)
    
    Args:
        model_layer: nn.Module (例如一个 nn.Linear)
        param_name: str (例如 'weight')
        input_data: Tensor, 这一层的输入数据 [Batch, ..., Input_Dim]
        layer_grad: Tensor, 反向传播计算出的梯度
        compression_rate: float,保留多少比例的参数 (例如 0.01 代表 1%)
    
    Returns:
        mask: Bool Tensor, 形状与 layer_grad 相同，True 表示保留
    """
    
    # 1. 准备随机投影矩阵 (Sketching Matrix)
    # 假设输出维度是 d_out，我们需要一个随机向量或小矩阵来降维
    # 这里简化：使用随机向量做投影，模拟 "Sketching"
    batch_size = input_data.shape[0]
    output_dim = model_layer.weight.shape[0] # Linear层的输出维度
    
    # 生成随机投影向量 v (形状要能和模型输出点积)
    # 这里的 v 相当于论文中的随机草图矩阵 R 的一列
    v = torch.randn(batch_size, output_dim, device=input_data.device)

    # 2. 定义计算函数：输入参数 -> 输出 -> 投影
    def f(params, x):
        # 使用 functional_call 进行无状态调用
        outputs = functional_call(model_layer, params, (x,))
        # 投影到随机向量上 (Project outputs)
        # 假设 outputs 是 [Batch, Out_Dim] (这里可能需要根据实际形状调整 reshape)
        if outputs.dim() > 2:
            outputs = outputs.reshape(batch_size, -1)[:, :output_dim] # 简化处理多维情况
        return (outputs * v).sum()

    # 3. 计算 NTK 梯度 (Jacobian Vector Product)
    # 这代表参数对输出变化的敏感度
    params = dict(model_layer.named_parameters())
    
    # 计算 output 关于权重的梯度 (这里的梯度就是 NTK 特征方向)
    # 使用 grad 计算标量输出的梯度
    ntk_grads = grad(f)(params, input_data)
    ntk_sensitivity = ntk_grads[param_name] # 获取权重的敏感度矩阵

    # 4. 结合：重要性得分 = |NTK敏感度| * |实际梯度|
    # 如果一个参数既在 NTK 主成分上 (敏感)，又有大梯度 (误差大)，它就是最重要的
    importance_score = torch.abs(ntk_sensitivity * layer_grad)

    # 5. 生成掩码 (Top-K 策略)
    k = int(layer_grad.numel() * compression_rate)
    threshold = torch.kthvalue(importance_score.view(-1), layer_grad.numel() - k).values
    mask = importance_score >= threshold
    
    return mask