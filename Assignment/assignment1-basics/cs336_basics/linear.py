import torch
from torch._higher_order_ops.out_dtype import out_dtype_fake_tensor_mode
import torch.nn as nn
import math

"""
    Linear层的作用:
        把一个vector的特征重新加权求和,映射到另一个特征空间中去
"""

class Linear(nn.Module):
    """核心: 1.维护一个weight参数 2.forward 过程计算矩阵乘法"""        
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # weigth : [d_dou,d_in]
        self.weight = nn.Parameter(
            torch.empty(out_features,in_features,device=device,dtype=dtype)
        )
        
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean = 0.0, std=std, a=-3 * std, b=3 * std)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x @ weights.T
        # x : [batch_size, seq_len, d_in]
        # self.weight.transpose(-1,-2) : [d_in, d_out]
        # [batch_size, seq_len, d_in] @ [d_in, d_out] -> [batch_size, seq_len, d_out]
        return x @ self.weight.transpose(-1,-2)

        