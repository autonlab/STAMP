import torch
import torch.nn as nn
import math
from stamp.modeling.acpe import ACPEAdapter
from stamp.modeling.gated_mlp import BasicGatedMLPBlock
from stamp.modeling.criss_cross_gated_mlp import CrissCrossGatedMLPBlock
from stamp.modeling.criss_cross_transformer import CrissCrossTransformerEncoder, CrissCrossTransformerEncoderLayer
from stamp.modeling.mhap import MultiHeadAttentionPooling

class STAMP(nn.Module):
    def __init__(
        self,
        input_dim,
        D,
        n_temporal_channels,
        n_spatial_channels,
        encoder_aggregation,
        n_classes,
        initial_proj_params,
        final_classifier_params,
        use_batch_norm,
        use_instance_norm,
        pe_params=None,
        transformer_params=None,
        gated_mlp_params=None,
        mhap_params=None,
        n_cls_tokens=None,
        ):
        super().__init__()
        assert input_dim > 0, "input_dim must be positive"
        assert D > 0, "D (model dimension) must be positive"
        assert n_temporal_channels > 0, "n_temporal_channels must be positive"
        assert n_spatial_channels > 0, "n_spatial_channels must be positive"
        assert encoder_aggregation in ['mean_across_tokens', 'attention_pooling', 'token_prediction_averaging'], \
                                    f"Unknown encoder_aggregation: {encoder_aggregation}"
        
        assert use_batch_norm + use_instance_norm < 2, 'use_batch_norm and use_instance_norm should not both be True.'

        self.D = D
        self.n_temporal_channels = n_temporal_channels
        self.n_spatial_channels = n_spatial_channels
        self.N = self.n_temporal_channels * self.n_spatial_channels
        self.pe_params = pe_params
        self.n_classes = n_classes
        self.n_cls_tokens = n_cls_tokens

        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        self.encoder_aggregation = encoder_aggregation
        self.gated_mlp_params = gated_mlp_params
        self.transformer_params = transformer_params

        if self.use_batch_norm:
            self.data_norm = nn.BatchNorm1d(input_dim)
        elif self.use_instance_norm:
            self.data_norm = nn.InstanceNorm1d(input_dim, affine=True)
        else:
            self.data_norm = None

        if self.D != input_dim:
            
            initial_proj_type = initial_proj_params['type']
            initial_proj_dropout_rate = initial_proj_params['dropout_rate']
            self.dropout = nn.Dropout(initial_proj_dropout_rate)

            if initial_proj_type == 'reduced':
                initial_proj_hidden_dim = initial_proj_params['hidden_dim']

                self.linear = nn.Sequential(
                    nn.Dropout(initial_proj_dropout_rate), # Input regularization
                    nn.Linear(input_dim, initial_proj_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(initial_proj_dropout_rate),
                    nn.Linear(initial_proj_hidden_dim, self.D)
                )
            elif initial_proj_type == 'full':
                self.linear = nn.Sequential(
                    nn.Dropout(initial_proj_dropout_rate),
                    nn.Linear(input_dim, self.D)
                )
            else:
                raise ValueError()

        if self.pe_params is not None:
            self.initialize_positional_embeddings()
            self.use_positional_embeddings = True
        else:
            self.use_positional_embeddings = False

        if self.n_cls_tokens is not None and self.n_cls_tokens > 0:
            self.cls_tokens = nn.Parameter(torch.randn(1, self.n_cls_tokens, self.D))

        # Initialize transformer
        if self.transformer_params is not None:
            if self.transformer_params['type'] == 'basic':
                self.initialize_basic_transformer()
            elif self.transformer_params['type'] == 'criss_cross':
                self.initialize_criss_cross_transformer()
            else:
                raise ValueError(f"Given transformer type, {self.transformer_params['type']}, is invalid.")

        # Initialize Gated MLP
        if self.gated_mlp_params is not None:
            gmlp_type = self.gated_mlp_params['type']
            n_layers = self.gated_mlp_params['n_layers']

            if self.gated_mlp_params['recurrent']:
                gmlp_block = self._create_gmlp_block(gmlp_type, self.gated_mlp_params)
                self.gated_mlp = nn.ModuleList([gmlp_block for _ in range(n_layers)])
            else:
                self.gated_mlp = nn.ModuleList([
                    self._create_gmlp_block(gmlp_type, self.gated_mlp_params)
                    for _ in range(n_layers)
                ])

        # Initialize encoder aggregation
        if self.encoder_aggregation == 'attention_pooling':
            assert mhap_params is not None, "MultiHeadAttentionPooling parameters must be provided for attention pooling."
            self.multi_head_attention_pooling = MultiHeadAttentionPooling(
                D=D,
                A=mhap_params['A'],
                dropout_rate=mhap_params['dropout_rate'],
                n_classes=self.n_classes,
                n_queries_per_head=mhap_params['n_queries_per_head'],
                query_combination=mhap_params['query_combination'],
                lambda_for_residual=mhap_params['lambda_for_residual']
                )
        elif self.encoder_aggregation == 'token_prediction_averaging':
            # Linear layer for token-level predictions
            self.token_classifier = nn.Linear(self.D, self.n_classes)
            self.multi_head_attention_pooling = None
        else:
            self.multi_head_attention_pooling = None
            self.linear_combiner = None
            self.token_classifier = None

        # Initialize final classifier
        if self.encoder_aggregation != 'token_prediction_averaging':
            if final_classifier_params is not None:
                dims = [self.D]
                if final_classifier_params.get('hidden_sizes'):
                    dims += final_classifier_params['hidden_sizes']
                dropout_rate = final_classifier_params.get('dropout_rate', None)
                layers = []

                for i, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
                    layers += [nn.Linear(in_dim, out_dim), nn.ELU()]
                    # Only add dropout if this isn't the last hidden layer
                    if dropout_rate is not None and i < len(dims) - 2:
                        layers.append(nn.Dropout(dropout_rate))

                layers.append(nn.Linear(dims[-1], self.n_classes))
                self.classifier = nn.Sequential(*layers)
            else:
                self.classifier = nn.Linear(self.D, self.n_classes)
        else:
            self.classifier = None

    def initialize_positional_embeddings(self):
        if 'basic' in self.pe_params['pe_type']:
            self.use_token_positional_embeddings = self.pe_params['use_token_positional_embeddings']
            self.use_spatial_positional_embeddings = self.pe_params['use_spatial_positional_embeddings']
            self.use_temporal_positional_embeddings = self.pe_params['use_temporal_positional_embeddings']
            if self.use_token_positional_embeddings:
                self.pos_embed = nn.Embedding(self.n_temporal_channels * self.n_spatial_channels, self.D) # Shape: (n_temporal_channels * n_spatial_channels, D), A unique positional embedding for each token
            if self.use_spatial_positional_embeddings:
                self.spatial_embed = nn.Embedding(self.n_spatial_channels, self.D) # Shape: (n_spatial_channels, D), A unique embedding for each spatial channel
            if self.use_temporal_positional_embeddings:
                self.temporal_embed = nn.Embedding(self.n_temporal_channels, self.D) # Shape: (n_temporal_channels, D), A unique embedding for each temporal channel
        elif self.pe_params['pe_type'] == 'sinusoidal':
            self.use_token_positional_embeddings = self.pe_params['use_token_positional_embeddings']
            self.use_spatial_positional_embeddings = self.pe_params['use_spatial_positional_embeddings']
            self.use_temporal_positional_embeddings = self.pe_params['use_temporal_positional_embeddings']

            self.max_tokens = self.n_temporal_channels * self.n_spatial_channels
            self.max_spatial = self.n_spatial_channels
            self.max_temporal = self.n_temporal_channels
        elif 'acpe' in self.pe_params['pe_type']:
            self.acpe = ACPEAdapter(D=self.D, kt=self.n_temporal_channels, ks=self.n_spatial_channels)
        else:
            raise NotImplementedError(f"Positional embedding type {self.pe_params['pe_type']} not implemented yet.")

    def build_positional_embeddings(self, x, N, B, T, S):
        if self.pe_params['pe_type'] == 'basic':
            embeds = self.build_basic_pe(x, N, B, T, S)
        elif self.pe_params['pe_type'] == 'sinusoidal':
            embeds = []
            # Use sinusoidal embeddings (non-parametric)
            if self.use_token_positional_embeddings:
                pos_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)  # Shape: (B, N)
                pos_embeds = create_sinusoidal_embeddings(pos_ids, self.D, max_len=self.max_tokens)  # Shape: (B, N, D)
                embeds.append(pos_embeds)

            if self.use_spatial_positional_embeddings:
                spatial_ids_for_temporal_channel = torch.arange(S, device=x.device)  # Shape: (S,)
                spatial_ids = spatial_ids_for_temporal_channel.repeat(T)  # Shape: (N,)
                spatial_ids = spatial_ids.unsqueeze(0).expand(B, -1)  # Shape: (B, N)
                spatial_embeds = create_sinusoidal_embeddings(spatial_ids, self.D, max_len=self.max_spatial)  # Shape: (B, N, D)
                embeds.append(spatial_embeds)

            if self.use_temporal_positional_embeddings:
                temporal_ids = torch.arange(T, device=x.device).repeat_interleave(S) # Shape: (N,)
                temporal_ids = temporal_ids.unsqueeze(0).expand(B, -1) # Shape: (B, N)
                temporal_embeds = create_sinusoidal_embeddings(temporal_ids, self.D, max_len=self.max_temporal)  # Shape: (B, N, D)
                embeds.append(temporal_embeds)
            embeds = torch.stack(embeds, dim=1).sum(dim=1)
        elif self.pe_params['pe_type'] == 'acpe':
            embeds = self.acpe(x, kt=self.pe_params['kt'], ks=self.pe_params['ks'], use_mix=self.pe_params['use_mix'])
        elif self.pe_params['pe_type'] == 'basic_and_acpe':
            basic_embeds = self.build_basic_pe(x, N, B, T, S)
            acpe_embeds = self.acpe(x, kt=self.pe_params['kt'], ks=self.pe_params['ks'], use_mix=self.pe_params['use_mix'])
            embeds = [basic_embeds, acpe_embeds]
            embeds = torch.stack(embeds, dim=1).sum(dim=1)
        else:
            raise NotImplementedError(f"Positional embedding type {self.pe_params['pe_type']} not implemented yet.")

        return embeds
    
    def build_basic_pe(self, x, N, B, T, S):
        embeds = []
        # Add learnable positional embeddings to each hour token (channels are embedding dims)
        if self.use_token_positional_embeddings:
            pos_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)  # Shape: (B, N)
            pos_embeds = self.pos_embed(pos_ids)  # Shape: (B, N, D)
            embeds.append(pos_embeds)

        if self.use_spatial_positional_embeddings:
            spatial_ids_for_temporal_channel = torch.arange(S, device=x.device)  # Shape: (S,)

            # Repeat for each hour
            spatial_ids = spatial_ids_for_temporal_channel.repeat(T)  # Shape: (N,)

            # Expand to batch
            spatial_ids = spatial_ids.unsqueeze(0).expand(B, -1)  # Shape: (B, N)

            spatial_embeds = self.spatial_embed(spatial_ids)  # Shape: (B, N, D)
            embeds.append(spatial_embeds)

        if self.use_temporal_positional_embeddings:
            temporal_ids = torch.arange(T, device=x.device).repeat_interleave(S) # Shape: (N,)

            # Expand to batch
            temporal_ids = temporal_ids.unsqueeze(0).expand(B, -1) # Shape: (B, N)
            temporal_embeds = self.temporal_embed(temporal_ids)  # Shape: (B, N, D)
            embeds.append(temporal_embeds)

        embeds = torch.stack(embeds, dim=1).sum(dim=1)
        return embeds
    
    def _create_gmlp_block(self, type, params):
        if type == 'basic':
            return BasicGatedMLPBlock(dim=self.D, dim_ff=params['dim_feedforward'], seq_len=self.N, 
                                        dropout_rate=params['dropout_rate'])
        elif type == 'criss_cross':
            return CrissCrossGatedMLPBlock(dim=self.D, dim_ff=params['dim_feedforward'], T=self.n_temporal_channels, 
                                            S=self.n_spatial_channels, combination_mode=params['combination_mode'], dropout_rate=params['dropout_rate'])
        else:
            raise ValueError(f"Given gated MLP type, {type}, is invalid.")
        
    def initialize_basic_transformer(self):
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.D,
                nhead=self.transformer_params['n_heads'],
                dim_feedforward=self.transformer_params['dim_feedforward'],
                dropout=self.transformer_params['dropout_rate'],
                activation='relu',
                batch_first=True,
                norm_first=self.transformer_params['norm_first'],
            )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.transformer_params['n_layers'],
            norm=nn.LayerNorm(self.D) if self.transformer_params.get('use_final_norm', True) else None
        )

    def initialize_criss_cross_transformer(self):
        """Initialize the criss-cross transformer encoder"""
        # Validate parameters
        assert self.D % 2 == 0, "Model dimension D must be even for criss-cross attention"
        assert self.transformer_params['n_heads'] % 2 == 0, "Number of heads must be even for criss-cross attention"

        encoder_layer = CrissCrossTransformerEncoderLayer(
            d_model=self.D,
            nhead=self.transformer_params['n_heads'],
            dim_feedforward=self.transformer_params['dim_feedforward'],
            dropout=self.transformer_params.get('dropout_rate', 0.1),
            activation=self.transformer_params.get('activation', 'relu'),
            batch_first=True,
            norm_first=self.transformer_params.get('norm_first', False)
        )

        # Set the required attributes for reshaping
        encoder_layer.n_temporal_channels = self.n_temporal_channels
        encoder_layer.n_spatial_channels = self.n_spatial_channels

        self.transformer = CrissCrossTransformerEncoder(
            encoder_layer,
            num_layers=self.transformer_params['n_layers'],
            norm=nn.LayerNorm(self.D) if self.transformer_params.get('use_final_norm', True) else None
        )

    def forward(self, x, return_attention):
        # x shape: (B, T, S, moment_embedding_dim) where the batch dimension represents patients
        B, T, S, input_dim = x.shape

        N = T * S

        if self.use_batch_norm:
            x = self.apply_batch_norm(x, B, T, S, input_dim)
        elif self.use_instance_norm:
            x = self.apply_instance_norm(x, B, T, S, input_dim)

        if input_dim != self.D:
            # Only apply linear transformation if the input dimension is different from D
            x = self.linear(x) # Shape: (B, T, S, D)

        tokens = x.reshape(B, N, self.D)  # Reshape to (B, N, D)

        if self.use_positional_embeddings:
            embeds = self.build_positional_embeddings(x, N, B, T, S)  # List of positional embeddings

            tokens = tokens + embeds  # Shape: (B, N, D)

        if self.gated_mlp_params is not None:
            if self.gated_mlp_params['type'] == 'criss_cross':
                tokens = tokens.reshape(B, T, S, self.D)

            # Apply the gated MLP blocks
            for layer in self.gated_mlp:
                tokens = layer(tokens) # Shape: (B, T, S, D)

            if self.gated_mlp_params['type'] == 'criss_cross':
                tokens = tokens.reshape(B, N, self.D)  # Reshape to (B, N, D)

        if self.transformer_params is not None:
            tokens = self.transformer(tokens, src_key_padding_mask=None)  # Shape: (B, N, D)

        if self.encoder_aggregation == 'mean_across_tokens':
            out = tokens.mean(dim=1)  # Shape: (B, D)
        elif self.encoder_aggregation == 'token_prediction_averaging':
            # Apply token-level classification and average predictions
            token_predictions = self.token_classifier(tokens)  # Shape: (B, N, n_classes)
            out = token_predictions.mean(dim=1)  # (B, n_classes)
        elif self.encoder_aggregation == 'attention_pooling':
            out, attn_weights = self.multi_head_attention_pooling(out=tokens, B=B)
        else:
            raise ValueError(f"Unknown encoder_aggregation method: {self.encoder_aggregation}")

        if self.encoder_aggregation != 'token_prediction_averaging':
            out = self.classifier(out) # Shape: (B, n_classes)

        if return_attention:
            return out, attn_weights.detach().cpu() if self.encoder_aggregation == 'attention_pooling' else None
        else:
            return out, None
    
    def apply_batch_norm(self, x, B, T, S, input_dim):
        # Reshape for batch norm: (B, T, S, input_dim) -> (B*T*S, input_dim)
        x = x.reshape(B * T * S, input_dim)
        x = self.data_norm(x)  # Apply data norm
        x = x.reshape(B, T, S, input_dim)  # Reshape back
        return x
    
    def apply_instance_norm(self, x, B, T, S, input_dim):
        # Reshape for instance norm: (B, T, S, input_dim) -> (B, input_dim, T*S)
        x = x.permute(0, 3, 1, 2).reshape(B, input_dim, T * S)
        x = self.data_norm(x)  # (B, input_dim, T*S)
        x = x.reshape(B, input_dim, T, S).permute(0, 2, 3, 1)  # Back to (B, T, S, input_dim)
        return x

def create_sinusoidal_embeddings(positions, d_model, max_len=10000):
    """
    Create sinusoidal positional embeddings

    Args:
        positions: tensor of positions, shape (batch_size, seq_len) or (seq_len,)
        d_model: embedding dimension
        max_len: maximum length for frequency calculation

    Returns:
        embeddings: tensor of shape (batch_size, seq_len, d_model) or (seq_len, d_model)
    """
    if positions.dim() == 1:
        positions = positions.unsqueeze(0)  # Add batch dimension if needed

    batch_size, seq_len = positions.shape

    # Create embedding matrix
    embeddings = torch.zeros(batch_size, seq_len, d_model, device=positions.device)

    # Calculate frequency terms
    div_term = torch.exp(torch.arange(0, d_model, 2, device=positions.device).float() *
                        -(math.log(max_len) / d_model))

    # Apply sin to even indices and cos to odd indices
    positions_expanded = positions.unsqueeze(-1).float()  # (batch_size, seq_len, 1)

    embeddings[:, :, 0::2] = torch.sin(positions_expanded * div_term)  # Even indices
    if d_model % 2 == 1:
        embeddings[:, :, 1::2] = torch.cos(positions_expanded * div_term[:-1])  # Odd indices (handle odd d_model)
    else:
        embeddings[:, :, 1::2] = torch.cos(positions_expanded * div_term)  # Odd indices

    return embeddings