import torch 
import torch.nn as nn 
import einops
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
    
def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device if device else torch.device("cpu")
        omg = 1.0 / (self.theta ** (torch.arange(0, d_k, 2).float() / d_k)) #(2k-2)/d
        self.register_buffer("omg", omg)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "... (d r) -> ... d r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return einops.rearrange(x, "... d r -> ... (d r)")
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape [..., seq_len, d_k]
        token_positions: tensor of shape [..., seq_len], typically just arange(seq_len)
        """
        seq_len = x.size(-2)
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device) # shape: (seq_len,)
            token_positions = token_positions.unsqueeze(0).expand(x.size(0), -1) # unsqueeze: (1, seq_len), expand to (batch_size, seq_len)

        
        theta = torch.einsum("... n, d ->  ... n d", token_positions, self.omg)
        
        # get pair cos and sine 
        cos = theta.cos().repeat_interleave(2, dim=-1)
        sin = theta.sin().repeat_interleave(2, dim=-1)
        
        x = x * cos + self._rotate_half(x) * sin

        return x

def Softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True).values
    #  subtract the largest entry of o_i from all elements of o_i, making the new largest entry 0
    x = torch.exp(x)
    x = x / torch.sum(x, dim=dim, keepdim=True)
    return x

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = k.size(-1)
    qk = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        qk = qk.masked_fill(mask == 0, float('-inf'))

    return torch.matmul(Softmax(qk), v)
        
   
