import torch
import torch.nn as nn

class Rope(nn.Module):
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        assert d_k % 2 == 0
        
        # k -> [1,..,d/2]
        # shape: (d_k // 2,)
        inv_freq = 1.0 / (theta **(torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k))
        
        # 位置索引
        # shape: (max_seq_len,)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        
        # 外积得到每个位置、每个频率对应的角度
        # shape: (max_seq_len, d_k // 2)
        freqs = torch.outer(positions, inv_freq)
        
        # 注意：register_buffer 不要赋值回成员变量
        self.register_buffer("cos_cached",torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached",torch.sin(freqs), persistent=False)
        
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2] # (..., seq_len, d_k // 2)
        x_odd = x[..., 1::2] # (..., seq_len, d_k // 2)
        
        
        # 根据 token_positions 取出对应位置的 cos / sin
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        # 应用二维旋转
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        
        
        # 把 even / odd 交错还原回最后一维
        out = torch.empty_like(x)
        out[...,0::2] = out_even
        out[...,1::2] = out_odd
        
        return out
    
    
    