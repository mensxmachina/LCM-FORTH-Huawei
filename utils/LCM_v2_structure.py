# model specification class for the second version of LCM models

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LCM_v2(nn.Module):
    def __init__(
        self,
        n_vars=12,
        max_lag=3,
        max_seq_len=500,
        d_model=32,
        n_heads=4,
        n_blocks=1,
        d_ff=64,
        dropout_coeff=0.1,
        patch_len=16,
        stride=4,
        attention_distilation=True,
        training_aids=False,
        **kwargs
        ):
        super().__init__()

        self.n_vars = n_vars
        self.max_lag = max_lag
        self.training_aids = training_aids
        self.d_model = d_model
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (max_seq_len - patch_len) // stride + 1

        self.patching = Patching(patch_len=patch_len, stride=stride)
        self.patch_embedding = PatchEmbedding(patch_len=patch_len, d_model=d_model)
        self.pos_embedding = PositionalEmbedding(d_model)

        self.encoder = nn.Sequential(
            *[EncoderLayer(d_model, n_heads, dropout_coeff) for _ in range(n_blocks)]
        )

        # Output layers
        in_dim = d_model * n_vars
        if self.training_aids:
            in_dim = self.n_vars * self.d_model + (self.n_vars * self.n_vars * self.max_lag)
        else:
            in_dim = self.n_vars * self.d_model

        self.norm = nn.LayerNorm(d_ff)
        self.activation = SwiGLU(dim=d_ff, hidden_dim=d_ff)
        self.fc1 = nn.Linear(in_dim, d_ff)
        self.fc2 = nn.Linear(d_ff, n_vars**2 * max_lag)

        self.dropout = nn.Dropout(dropout_coeff)

    def forward(self, x):
        # Handle input types
        corr, te = None, None
        if isinstance(x, (list, tuple)):
            if self.training_aids:
                if len(x) < 2:
                    raise ValueError("Expected data and cross-correlation input.")
                x, corr = x[:2]
            else:
                x = x[0] if len(x) > 0 else x

        batch_size = x.size(0)

        x = self.patching(x) # [B*N, num_patches, patch_len]
        x = self.patch_embedding(x) # [B*N, num_patches, D]
        P = x.shape[1] # num_patches

        x = x.view(batch_size, self.n_vars, P, -1).permute(0, 2, 1, 3).contiguous()  # [B, P, N, D]

        x = self.encoder(x)  # [B, P, N, D]

        x = x.mean(dim=1)  # Aggregate over patches (dim=1), [B, N, D]
        x = x.reshape(batch_size, -1)  # [B, N * D] 

        if self.training_aids:
            if corr is None:
                raise ValueError("training_aids=True, but `corr` is not provided.")
            #if te is None:
            #    raise ValueError("training_aids=True, but `te` is not provided.")

            corr = corr.contiguous().view(batch_size, -1)

            x = torch.cat((x, corr), dim=1)

        x = self.activation(self.norm(self.fc1(x)))  # [B, d_ff]
        x = self.fc2(x)  # [B, n_vars^2 * max_lag]

        return x.reshape(batch_size, self.n_vars, self.n_vars, self.max_lag)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int=16, max_length: int=10000):
        """
        Sinusoidal positional embeddings as per Vaswani et al, 2017.
        """
        super(PositionalEmbedding, self).__init__()
        embedding = torch.zeros(max_length, d_model) # shape is [max_length, d_model]

        position = torch.arange(0, max_length).float().unsqueeze(1) # shape is [max_length, 1]
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        embedding[:, 0::2] = torch.sin(position * div_term) # even terms
        embedding[:, 1::2] = torch.cos(position * div_term) # odd terms

        embedding = embedding.unsqueeze(0) # [1, max_length, d_model]
        self.register_buffer("embedding", embedding)

    def forward(self, x):
        return self.embedding[:, : x.size(1)] # forward pass is of shape [1, x.size(1), d_model], just returns the positional encoding

class Patching(nn.Module):
    def __init__(self, patch_len, stride):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        """
        Input: [B, T, V]
        Output: [B * V, num_patches, patch_len]
        """
        B, T, V = x.shape
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # [B, num_patches, V, patch_len]
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, V, num_patches, patch_len]
        x = x.view(B * V, x.shape[2], self.patch_len)  # [B*V, num_patches, patch_len]

        return x
    
class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, d_model, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(patch_len, d_model)
        self.pos_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Input: [B * V, num_patches, patch_len]
        Output: [B * V, num_patches, d_model]
        """
        x = self.linear(x)  # [B*V, num_patches, d_model]
        x = x + self.pos_embedding(x)  # [B*V, num_patches, d_model]

        return self.dropout(x)

class TemporalAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        # x: [B, P, N, D]
        B, P, N, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B * N, P, D)  # [B*N, P, D]
        x, _ = self.attn(x, x, x)  # Temporal attention
        x = x.view(B, N, P, D).permute(0, 2, 1, 3)  # [B, P, N, D]
        return x

class SpatialAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        # x: [B, P, N, D]
        B, P, N, D = x.shape
        x = x.permute(0, 1, 2, 3).contiguous().view(B * P, N, D)  # [B*P, N, D]
        x, _ = self.attn(x, x, x)  # Spatial attention
        x = x.view(B, P, N, D)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.temporal_attn = TemporalAttention(d_model, n_heads, dropout)
        self.spatial_attn = SpatialAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        # x: [B, P, N, D]
        x = self.temporal_attn(x) + x
        x = self.dropout(x) # testing
        x = self.norm1(x)
        x = self.spatial_attn(x) + x
        x = self.dropout(x) # testing
        x = self.norm2(x)
        x = self.ff(x) + x
        return x

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        """
        SwiGLU (Swish Gated Linear Unit) activation layer
        
        Args:
        ---
            dim (int): input dimension
            hidden_dim (int): hidden dimension (typically 4x input dim)
            
        Notes:
        ---
        [*] Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202
        """
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x):
        """
        Forward pass for SwiGLU activation
        
        Args:
        ---
            x (torch.tensor): input tensor of shape [..., dim] 
            
        Returns:
        ---
            (torch.tensor): output tensor of shape [..., dim]
        """
        x, gate = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * x)
