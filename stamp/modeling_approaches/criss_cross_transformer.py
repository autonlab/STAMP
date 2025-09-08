import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Optional, Callable, Union

'''
This implementation is directly adapted from CBraMod:
https://github.com/wjq-learning/CBraMod/blob/main/models/criss_cross_transformer.py
'''
class CrissCrossTransformerEncoderLayer(nn.Module):
    """
    Criss-Cross Transformer Encoder Layer adapted for temporal-spatial data
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: Union[str, Callable] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 norm_first: bool = False, bias: bool = True, device=None, dtype=None):
        super().__init__()

        # Split attention heads between spatial and temporal
        assert nhead % 2 == 0, "nhead must be even for criss-cross attention"
        assert d_model % 2 == 0, "d_model must be even for criss-cross attention"

        # Spatial attention: attention across spatial channels at each temporal position
        self.spatial_attn = nn.MultiheadAttention(
            d_model // 2, nhead // 2, dropout=dropout,
            bias=bias, batch_first=batch_first
        )

        # Temporal attention: attention across temporal positions for each spatial channel
        self.temporal_attn = nn.MultiheadAttention(
            d_model // 2, nhead // 2, dropout=dropout,
            bias=bias, batch_first=batch_first
        )

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Handle activation function
        if isinstance(activation, str):
            activation = self._get_activation_fn(activation)
        self.activation = activation

    def _get_activation_fn(self, activation: str):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        else:
            raise RuntimeError(f"activation should be relu/gelu, not {activation}")

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: Input tensor of shape (B, N, D) where N = T * S
            src_mask: Optional attention mask
            src_key_padding_mask: Optional padding mask of shape (B, N)
        """
        x = src
        if self.norm_first:
            x = x + self._criss_cross_attention_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._criss_cross_attention_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _criss_cross_attention_block(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor],
                                   key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply criss-cross attention to input tensor

        Args:
            x: Input tensor of shape (B, N, D) where N = T * S

        The input x was created by: original_x.reshape(B, N, D) where original_x had shape (B, T, S, D)
        This means the flattening order is: all S channels for T=0, then all S channels for T=1, etc.
        So token index i corresponds to: h = i // S, c = i % S
        """
        B, N, D = x.shape

        # We need to know T and S to reshape properly
        T = getattr(self, 'n_temporal_channels', None)
        S = getattr(self, 'n_spatial_channels', None)

        if T is None or S is None:
            raise ValueError("n_temporal_channels and n_spatial_channels must be set")

        assert N == T * S, f"Expected N={T*S}, got N={N}"

        # Split features into two halves
        x_spatial = x[:, :, :D//2]  # (B, N, D/2)
        x_temporal = x[:, :, D//2:]  # (B, N, D/2)

        # SPATIAL ATTENTION: attend across spatial channels at each temporal position
        # Since original flattening was (B, T, S, D) -> (B, T*S, D), we can reshape back directly
        # x_spatial: (B, N, D/2) -> (B, T, S, D/2) -> (B*T, S, D/2)
        x_spatial_reshaped = x_spatial.view(B, T, S, D//2).contiguous().view(B*T, S, D//2)

        # Create padding mask for spatial attention if provided
        spatial_padding_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, N) -> (B, T, S) -> (B*T, S)
            spatial_padding_mask = key_padding_mask.view(B, T, S).view(B*T, S)

        # Apply spatial attention: each temporal position attends across its spatial channels
        x_spatial_attn, _ = self.spatial_attn(
            x_spatial_reshaped, x_spatial_reshaped, x_spatial_reshaped,
            attn_mask=attn_mask, key_padding_mask=spatial_padding_mask,
            need_weights=False
        )

        # Reshape back: (B*T, S, D/2) -> (B, T, S, D/2) -> (B, N, D/2)
        x_spatial_out = x_spatial_attn.view(B, T, S, D//2).contiguous().view(B, N, D//2)

        # TEMPORAL ATTENTION: attend across temporal positions for each spatial channel
        # We need to group by spatial channel, so we need to transpose T and S dimensions
        # x_temporal: (B, N, D/2) -> (B, T, S, D/2) -> (B, S, T, D/2) -> (B*S, T, D/2)
        x_temporal_reshaped = x_temporal.view(B, T, S, D//2).permute(0, 2, 1, 3).contiguous().view(B*S, T, D//2)

        # Create padding mask for temporal attention if provided
        temporal_padding_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, N) -> (B, T, S) -> (B, S, T) -> (B*S, T)
            temporal_padding_mask = key_padding_mask.view(B, T, S).permute(0, 2, 1).contiguous().view(B*S, T)

        # Apply temporal attention: each spatial channel attends across temporal positions
        x_temporal_attn, _ = self.temporal_attn(
            x_temporal_reshaped, x_temporal_reshaped, x_temporal_reshaped,
            attn_mask=attn_mask, key_padding_mask=temporal_padding_mask,
            need_weights=False
        )

        # Reshape back: (B*S, T, D/2) -> (B, S, T, D/2) -> (B, T, S, D/2) -> (B, N, D/2)
        x_temporal_out = x_temporal_attn.view(B, S, T, D//2).permute(0, 2, 1, 3).contiguous().view(B, N, D//2)

        # Concatenate spatial and temporal attention outputs
        x_out = torch.cat([x_spatial_out, x_temporal_out], dim=-1)  # (B, N, D)

        return self.dropout1(x_out)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class CrissCrossTransformerEncoder(nn.Module):
    """
    Criss-Cross Transformer Encoder
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output