import torch.nn as nn
import torch
import math

class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, D, A, dropout_rate, n_classes, n_queries_per_head, query_combination, lambda_for_residual):
        super().__init__()
        assert A is not None, "n_heads must be specified for multi-head attention pooling."
        self.D = D # Dimension of the input tokens
        self.A = A # Number of attention heads
        self.d = D // A # Dimension of each attention head
        assert D % A == 0, "D must be divisible by A"
        self.dropout_rate = dropout_rate
        self.n_classes = n_classes
        self.K = n_queries_per_head
        self.query_combination = query_combination  # 'max', 'weighted_sum', 'learned_weights', 'all_concat'
        self.lambda_for_residual = lambda_for_residual

        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)

        self.W = nn.ModuleList([nn.Linear(self.D, self.d, bias=False) for _ in range(self.A)]) # Separate projection matrices for each head, Shape: (A, D, d)

        self.Q = nn.Parameter(torch.randn(A, self.K, self.d))  # Shape: (A, K, d)

        # Different combination strategies
        if query_combination == 'learned_weights':
            # Learn to weight different queries based on input
            self.query_weight_net = nn.Sequential(
                nn.Linear(D, D // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate if dropout_rate is not None else 0.0),
                nn.Linear(D // 2, A * self.K),
                nn.Softmax(dim=-1)
            )
        elif query_combination == 'all_concat':
            # Use all queries - increase output dimension
            self.out_proj = nn.Linear(D * self.K, D)
        else:
            self.out_proj = nn.Linear(D, D)

        if query_combination != 'all_concat':
            self.out_proj = nn.Linear(D, D) # Output projection

        self.softmax = nn.Softmax(dim=3) # Normalize over sequence (T dimension with K queries)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters for stability"""
        # Initialize query vectors with small random values and encourage diversity
        with torch.no_grad():  # Disable gradient tracking for initialization
            for a in range(self.A):
                for k in range(self.K):
                    # Initialize each query differently to encourage diversity
                    nn.init.normal_(self.Q[a, k], std=0.02)
                    # Add small random rotation to encourage different directions
                    if k > 0:
                        self.Q[a, k].add_(0.1 * torch.randn_like(self.Q[a, k]))

        # Initialize projection matrices
        for w in self.W:
            nn.init.xavier_uniform_(w.weight)

        # Initialize learned weight network if it exists
        if hasattr(self, 'query_weight_net'):
            for layer in self.query_weight_net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Initialize output projection
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, out, B):
        residual = out.mean(dim=1) if self.lambda_for_residual > 0 else None  # For residual connection

        # 1. Project token outputs with dropout for regularization
        projections = []
        for i in range(self.A):
            proj = self.W[i](out)  # Shape: (B, T, d)
            if self.dropout_rate is not None:
                proj = self.dropout(proj)  # Add dropout to projections
            projections.append(proj)
        projections = torch.stack(projections, dim=1)  # Shape: (B, A, N, d)

        # 2. Compute attention scores for all queries
        queries = self.Q.unsqueeze(0).unsqueeze(3)  # Shape: (1, A, Q, 1, d)
        projections_exp = projections.unsqueeze(2)   # Shape: (B, A, 1, N, d)
        scores = (projections_exp * queries).sum(dim=-1) / math.sqrt(self.d) # Shape: (B, A, Q, N)

        # 3. Normalize attention scores using softmax
        attn_weights = self.softmax(scores)  # Shape: (B, A, Q, N)

        # Handle NaN values
        attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # Apply dropout to attention weights
        if self.dropout_rate is not None:
            attn_weights = self.dropout(attn_weights)

        # 4. Different query combination strategies
        if self.query_combination == 'max':
            # Original: select best query per head
            query_scores = attn_weights.sum(dim=-1)  # Shape: (B, A, K)
            best_query_idx = query_scores.argmax(dim=-1)  # Shape: (B, A)

            best_query_idx_exp = best_query_idx.unsqueeze(-1).unsqueeze(-1)  # Shape: (B, A, 1, 1)
            best_query_idx_exp = best_query_idx_exp.expand(-1, -1, -1, attn_weights.size(-1))  # Shape: (B, A, 1, T)
            selected_attn_weights = torch.gather(attn_weights, dim=2, index=best_query_idx_exp).squeeze(2)  # Shape: (B, A, T)

            attn_weights_for_return = selected_attn_weights.detach().cpu()
            selected_attn_weights = selected_attn_weights.unsqueeze(-1)  # Shape: (B, A, T, 1)

            # Compute weighted sum
            weighted_tokens = (selected_attn_weights * projections)  # Shape: (B, A, T, d)
            pooled = weighted_tokens.sum(dim=2)  # Shape: (B, A, d)

        elif self.query_combination == 'weighted_sum':
            # Weight queries by their attention strength, each query gets a weight based on its total attention across tokens
            query_scores = attn_weights.sum(dim=-1)  # Shape: (B, A, Q)
            query_weights = torch.softmax(query_scores, dim=-1)  # Shape: (B, A, Q), the query weights for each head sum to 1

            attn_weights_exp = attn_weights.unsqueeze(-1)  # Shape: (B, A, Q, N, 1)
            projections_exp_2 = projections.unsqueeze(2)  # Shape: (B, A, 1, N, d)

            # Weight tokens by their attention scores per query
            query_weighted_tokens = attn_weights_exp * projections_exp_2 # Shape: (B, A, Q, N, d)
            query_pooled = query_weighted_tokens.sum(dim=3)  # Shape: (B, A, Q, d), pooling over tokens

            # Weight the pooled representations by the query weights
            query_weights_exp = query_weights.unsqueeze(-1) # Shape: (B, A, Q, 1)
            pooled = (query_weights_exp * query_pooled).sum(dim=2)  # Shape: (B, A, d)

            # For visualization - use the highest weighted query's attention
            best_query_idx = query_weights.argmax(dim=-1) # (B, A)
            # Prepare index tensor for gather
            best_query_idx_exp = best_query_idx.unsqueeze(-1).unsqueeze(-1)  # (B, A, 1, 1)
            best_query_idx_exp = best_query_idx_exp.expand(-1, -1, 1, attn_weights.size(-1))  # (B, A, 1, T)

            attn_weights_for_return = torch.gather(attn_weights, dim=2, index=best_query_idx_exp)  # (B, A, 1, T)
            attn_weights_for_return = attn_weights_for_return.squeeze(2).detach().cpu()  # (B, A, T)

        elif self.query_combination == 'learned_weights':
            # Learn input-dependent weights for queries
            mean_repr = out.mean(dim=1)  # Shape: (B, D)
            query_weights = self.query_weight_net(mean_repr)  # Shape: (B, A * K)
            query_weights = query_weights.view(B, self.A, self.K)  # Shape: (B, A, K)

            # Same as weighted_sum but with learned weights
            attn_weights_exp = attn_weights.unsqueeze(-1)  # Shape: (B, A, K, T, 1)
            projections_exp_2 = projections.unsqueeze(2)  # Shape: (B, A, 1, T, d)

            query_weighted_tokens = attn_weights_exp * projections_exp_2
            query_pooled = query_weighted_tokens.sum(dim=3)  # Shape: (B, A, K, d)

            query_weights_exp = query_weights.unsqueeze(-1)
            pooled = (query_weights_exp * query_pooled).sum(dim=2)  # Shape: (B, A, d)

            # For visualization
            best_query_idx = query_weights.argmax(dim=-1)
            best_query_idx_exp = best_query_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, attn_weights.size(-1))
            attn_weights_for_return = torch.gather(attn_weights, dim=2, index=best_query_idx_exp).squeeze(2).detach().cpu()

        elif self.query_combination == 'all_concat':
            # Use all queries - concatenate their results
            attn_weights_exp = attn_weights.unsqueeze(-1)  # Shape: (B, A, K, T, 1)
            projections_exp_2 = projections.unsqueeze(2)  # Shape: (B, A, 1, T, d)

            query_weighted_tokens = attn_weights_exp * projections_exp_2
            query_pooled = query_weighted_tokens.sum(dim=3)  # Shape: (B, A, K, d)

            # Concatenate all queries for each head
            pooled = query_pooled.view(B, self.A, self.K * self.d)  # Shape: (B, A, K*d)

            # For visualization - use first query
            attn_weights_for_return = attn_weights[:, :, 0, :].detach().cpu()

        # 5. Final projection
        if self.query_combination == 'all_concat':
            # Need to handle concatenated dimension
            pooled = pooled.view(B, self.A * self.K * self.d)  # Shape: (B, A*K*d)
            out = self.out_proj(pooled)  # Shape: (B, D)
        else:
            # Standard case
            pooled = pooled.view(B, self.D)  # Shape: (B, D)
            out = self.out_proj(pooled)  # Shape: (B, D)

        # Optional residual connection for stability
        if self.lambda_for_residual > 0 and residual is not None:
            out = out + self.lambda_for_residual * residual

        return out, attn_weights_for_return