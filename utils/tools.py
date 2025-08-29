import torch
import torch.nn as nn
import math
# import einops

# Weighted MSE loss function 
class weighted_mse:
    def __init__(self, scaling=90):
        self.mse = nn.MSELoss(reduction="none")
        self.scaling = scaling

    def __call__(self, inp, target):
        # get target weight vector
        weights = torch.ones(target.shape, device=inp.device)
        weights[target > 0] = self.scaling
        return (weights * self.mse(inp, target)).mean() 
    
# Lagged batch cross-correlation
def lagged_batch_corr(points, max_lags):
    B, N, D = points.size()

    # roll the data to create lagged versions of them
    rolled_points = [torch.roll(points, shifts=x, dims=1) for x in range(max_lags + 1)]
    stack = torch.cat(rolled_points, dim=2)

    mean = stack.mean(dim=1).unsqueeze(1)
    std = stack.std(dim=1).unsqueeze(1) + 1e-6 # avoiding division by zero
    diffs = (stack - mean).reshape(B * N, D * (max_lags + 1)) # centering

    # cross-correlation matrix calculation
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(
        B, N, D * (max_lags + 1), D * (max_lags + 1)
    )

    bcov = prods.sum(dim=1) / (N - 1)  # unbiased estimate

    # normalize to obtain the final cross-correlation matrix
    corr = bcov / (
        std.repeat(1, D * (max_lags + 1), 1).reshape(
            std.shape[0], D * (max_lags + 1), D * (max_lags + 1)
        )
        * std.permute((0, 2, 1))
    )
    # remove backward time-lag links and thus return only forward cross-correlations
    return corr[:, :D, D:]  # (B, D, D)


def rearrange_corr(corr, n_vars, t):
    """
    Recreate einops.rearrange(corr[:, :n_vars], "b c1 (t c2) -> b c1 c2 t", t=t)
    
    Args:
        corr (torch.Tensor): Input tensor of shape (b, c1, t*c2)
        n_vars (int): Number of variables (c1 slice)
        t (int): Size of the 't' dimension
    
    Returns:
        torch.Tensor: Rearranged tensor of shape (b, c1, c2, t)
    """
    corr_sliced = corr[:, :n_vars]          # slice c1 dimension
    b, c1, _ = corr_sliced.shape
    c2 = corr_sliced.shape[2] // t          # infer c2
    return corr_sliced.view(b, c1, c2, t)   # reshape to (b, c1, c2, t)

# Helping function for the custom_corr_regularization function.
def transform_corr_to_y(corr, ml, n_vars):
    # ncorr = einops.rearrange(corr[:, :n_vars:], "b c1 (t c2) -> b c1 c2 t", t=ml)
    # assert einops.rearrange(corr[:, :n_vars:], "b c1 (t c2) -> b c1 c2 t", t=ml) == rearrange_corr(corr, n_vars=n_vars, t=ml)

    ncorr = rearrange_corr(corr, n_vars=n_vars, t=ml)
    fncorr = torch.flip(ncorr, dims=[3])
    return fncorr


# A helping function during inference. Exclusive only to models that have been trained with the 
# correlation regularization technique. Interested readers can find details here (Stein et al, 2024):
# https://github.com/Gideon-Stein/CausalPretraining
def custom_corr_regularization(predictions, data, exp=1.5, epsilon=0.15): 
    # - predictions: (batch, cause, effect, lag)
    # - data: batch, t, n_vars
    # - penalized predictions if the cov of the corresponding link is low.
    ml = predictions.shape[3]
    n_vars = data.shape[2]

    # rashape everything properly
    corr = lagged_batch_corr(data, ml)
    fncorr = transform_corr_to_y(corr, ml, n_vars)
    # specifying the batch size
    regularization = 1 / (torch.abs(fncorr) + epsilon)  # for numeric stability

    penalty = torch.mean((predictions * regularization) ** exp)
    return penalty


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


class LearnableEmbedding(nn.Module):
    def __init__(self, d_model: int=16, max_length: int=500):
        """
        Learnable positional embeddings, initialized as random tensors of shape (L, d_model)
        [*] Wang, Yu-An, and Yun-Nung Chen. "What do position embeddings learn? an empirical study of pre-trained language model positional encoding." 
        arXiv preprint arXiv:2010.04903 (2020).
        """
        super(LearnableEmbedding, self).__init__()
        embedding = nn.Parameter(torch.rand(max_length, d_model))

        embedding = embedding.unsqueeze(0)
        self.register_buffer("embedding", embedding)

    def forward(self, x):
        return self.embedding[:, : x.size(1)] # forward pass is of shape [1, x.size(1), d_model], just returns the positional embedding


class RelativePositionEmbedding(nn.Module):
    def __init__(self, num_units: int, max_relative_position: int):
        """
        Relative positional embeddings according to Shaw et al., 2018
        [*] Shaw, Peter, Jakob Uszkoreit, and Ashish Vaswani. "Self-attention with Relative Position Representations." arXiv preprint arXiv:1803.02155 (2018).
        """
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings