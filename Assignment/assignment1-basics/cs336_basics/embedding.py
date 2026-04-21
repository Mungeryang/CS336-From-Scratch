import torch
import torch.nn as nn

from cs336_basics.linear import Linear

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.nun_embeddings = num_embeddings
        self.d_model = embedding_dim
        
        self.weigths = nn.Parameter(
            torch.empty(self.nun_embeddings,self.d_model,device=device,dtype=dtype)
        )
        
        nn.init.trunc_normal_(self.weigths, mean=0.0, std=1,a = -3,b = 3)
        
        
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weigths[token_ids]