import torch.nn as nn

class BasicGatedMLPBlock(nn.Module):
    """
    Gated MLP block from "Pay Attention to MLPs" paper.
    
    Args:
        dim: Model dimension
        dim_ff: Feed-forward dimension (typically 4 * dim)
        seq_len: Sequence length for spatial projections
        dropout_rate: Dropout probability
    """
    def __init__(self, dim, dim_ff, seq_len, dropout_rate=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj_1 = nn.Linear(dim, dim_ff)
        self.gelu = nn.GELU()
        self.sgu = SpatialGatingUnit(dim_ff, seq_len)
        self.proj_2 = nn.Linear(dim_ff // 2, dim)  # Note: dim_ff//2 because SGU outputs half
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        # x shape: (B, N, D)
        shortcut = x
        x = self.norm(x)  # Layer normalization, (B, N, D)
        x = self.proj_1(x)   # Project to dim_ff, (B, N, dim_ff)
        x = self.gelu(x)   # GELU activation, (B, N, dim_ff)
        x = self.sgu(x)   # Spatial gating unit, (B, N, dim_ff//2)
        x = self.proj_2(x)   # Project back to D, (B, N, D)
        x = self.dropout(x) # (B, N, D)
        return x + shortcut # (B, N, D)
    
class SpatialGatingUnit(nn.Module):
    def __init__(self, dim_ff, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(dim_ff // 2)
        self.spatial_proj = nn.Linear(seq_len, seq_len)

        # Initialize weights to small values near 0
        nn.init.normal_(self.spatial_proj.weight, mean=0.0, std=1e-6)
        # Initialize bias to 1 as mentioned in paper
        nn.init.constant_(self.spatial_proj.bias, 1.0)
    
    def forward(self, x):
        u, v = x.chunk(2, dim=-1)  # Split embedding dimension into two chunks, (B, N, dim_ff//2) and (B, N, dim_ff//2)
        v = self.norm(v)  # Normalize v along embedding dimension
        # Apply spatial projection along token dimension
        v = v.transpose(-1, -2) # (B, N, dim_ff//2) -> (B, dim_ff//2, N) 
        v = self.spatial_proj(v) # Mix across token dimension
        v = v.transpose(-1,-2) # (B, dim_ff//2, N) -> (B, N, dim_ff//2)
        return u * v  # Element-wise gating
