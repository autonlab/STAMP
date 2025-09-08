import torch
import torch.nn as nn

class CrissCrossGatedMLPBlock(nn.Module):
    """
    Dual-axis Gated MLP block for EEG data with shape (B, T, S, D).
    Applies both spatial and temporal mixing to the full feature representation.
    
    Args:
        dim: Embedding dimension (D)
        dim_ff: Feed-forward dimension (typically 4 * dim)
        T: Number of temporal channels
        S: Number of spatial channels
        dropout_rate: Dropout probability
    """
    def __init__(self, dim, dim_ff, T, S, combination_mode, dropout_rate=0.0):
        super().__init__()

        self.combination_mode = combination_mode
        
        self.norm = nn.LayerNorm(dim)
        self.proj_1 = nn.Linear(dim, dim_ff)
        self.gelu = nn.GELU()
        
        # Both SGUs work on the full feature representation
        self.sgu_temporal = SpatialGatingUnit(dim_ff, T, axis='temporal')
        self.sgu_spatial = SpatialGatingUnit(dim_ff, S, axis='spatial')
        
        # Set up final projection based on combine mode
        if combination_mode == 'concat':
            self.proj_2 = nn.Linear(dim_ff, dim)  # dim_ff//2 + dim_ff//2 = dim_ff
        elif combination_mode in ['add', 'weighted_add']:
            self.proj_2 = nn.Linear(dim_ff // 2, dim)  # More efficient projection
        else:
            raise ValueError(f"Invalid combination_mode: {combination_mode}. Must be 'concat', 'add', or 'weighted_add'")
        
        if combination_mode == 'weighted_add':
            self.temporal_weight = nn.Parameter(torch.ones(1))
            self.spatial_weight = nn.Parameter(torch.ones(1))
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        # x shape: (B, T, S, D)
        shortcut = x
        x = self.norm(x)  # (B, T, S, D)
        x = self.proj_1(x)  # (B, T, S, dim_ff)
        x = self.gelu(x)  # (B, T, S, dim_ff)
        
        # Apply both SGUs to the same input
        temporal_out = self.sgu_temporal(x)  # (B, T, S, dim_ff//2)
        spatial_out = self.sgu_spatial(x)   # (B, T, S, dim_ff//2)
        
        # Combine outputs from both mixing operations
        # Combine outputs based on mode
        if self.combination_mode == 'concat':
            x = torch.cat([temporal_out, spatial_out], dim=-1)  # (B, T, S, dim_ff)
        elif self.combination_mode == 'add':
            x = temporal_out + spatial_out  # (B, T, S, dim_ff//2)
        elif self.combination_mode == 'weighted_add':
            x = self.temporal_weight * temporal_out + self.spatial_weight * spatial_out  # (B, T, S, dim_ff//2)
        
        x = self.proj_2(x)  # (B, T, S, D)
        x = self.dropout(x)
        return x + shortcut

class SpatialGatingUnit(nn.Module):
    """
    Generalized Spatial Gating Unit that can mix along different axes.
    
    Args:
        dim_ff: Feed-forward dimension
        mix_dim: Dimension size to mix across
        axis: Which axis to mix ('spatial' for channels, 'temporal' for time)
    """
    def __init__(self, dim_ff, mix_dim, axis='spatial'):
        super().__init__()
        self.axis = axis
        self.norm = nn.LayerNorm(dim_ff // 2)
        self.spatial_proj = nn.Linear(mix_dim, mix_dim)

        # Initialize weights to small values near 0
        nn.init.normal_(self.spatial_proj.weight, mean=0.0, std=1e-6)
        # Initialize bias to 1 as mentioned in paper
        nn.init.constant_(self.spatial_proj.bias, 1.0)
    
    def forward(self, x):
        # x shape: (B, T, S, dim_ff)
        u, v = x.chunk(2, dim=-1)  # Split along embedding dimension: (B, T, S, dim_ff//2)
        v = self.norm(v)  # Normalize v along embedding dimension
        
        if self.axis == 'temporal':
            # Mix across temporal dimension (time)
            # Move T to last position: (B, T, S, dim_ff//2) -> (B, S, dim_ff//2, T)
            v = v.permute(0, 2, 3, 1)  # (B, T, S, dim_ff//2) -> (B, S, dim_ff//2, T)
            v = self.spatial_proj(v)  # Mix across T dimension
            v = v.permute(0, 3, 1, 2)  # (B, S, dim_ff//2, T) -> (B, T, S, dim_ff//2)
            
        elif self.axis == 'spatial':
            # Mix across spatial dimension (channels)
            # Move S to last position: (B, T, S, dim_ff//2) -> (B, T, dim_ff//2, S)
            v = v.permute(0, 1, 3, 2)  # (B, T, S, dim_ff//2) -> (B, T, dim_ff//2, S)
            v = self.spatial_proj(v)  # Mix across S dimension
            v = v.permute(0, 1, 3, 2)  # (B, T, dim_ff//2, S) -> (B, T, S, dim_ff//2)
        
        return u * v  # Element-wise gating
