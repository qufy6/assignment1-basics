import torch 
import torch.nn as nn 
from einops import einsum

class Linear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(in_features, out_features, device=device, dtype=dtype))
        self._init_weight()

    def _init_weight(self):
        # Parameter Initialization
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            x, self.weight, 
            "... d_in, d_in d_out -> ... d_out"
        )
    
class Embedding(nn.Module):
    def __init__(self, num_embedding: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embedding, embedding_dim, device = device, dtype = dtype))
    
    def _init_weight(self):
        # Parameter Initialization
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
        
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.empty(d_model, device = device, dtype = dtype))
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure x is float for numerical stability
        x = x.to(torch.float32)
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        normalized_x = x / norm

        return normalized_x * self.g
    


    
