import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    
    x_max = torch.max(x)
    
    x_trans = x - x_max
    
    exp_x = torch.exp(x_trans)
    
    exp_sum = torch.sum(exp_x, dim=dim, keepdim=True)
    
    out  = exp_x / exp_sum
    
    return out