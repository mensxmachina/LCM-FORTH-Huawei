import torch
import torch.nn as nn
import einops

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
    
# Lagged batch autocorrelation
def lagged_batch_corr(points, max_lags):
    B, N, D = points.size()

    # roll the data to create lagged versions of them
    rolled_points = [torch.roll(points, shifts=x, dims=1) for x in range(max_lags + 1)]
    stack = torch.cat(rolled_points, dim=2)

    mean = stack.mean(dim=1).unsqueeze(1)
    std = stack.std(dim=1).unsqueeze(1) + 1e-6 # avoiding division by zero
    diffs = (stack - mean).reshape(B * N, D * (max_lags + 1)) # centering

    # autocovariance matrix calculation
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(
        B, N, D * (max_lags + 1), D * (max_lags + 1)
    )

    bcov = prods.sum(dim=1) / (N - 1)  # unbiased estimate

    # normalize the covariance to obtain the final autocorrelation matrix
    corr = bcov / (
        std.repeat(1, D * (max_lags + 1), 1).reshape(
            std.shape[0], D * (max_lags + 1), D * (max_lags + 1)
        )
        * std.permute((0, 2, 1))
    )
    # remove backward time-lag links and thus return only forward autocorrelations
    return corr[:, :D, D:]  # (B, D, D)


# Helping function for the custom_corr_regularization function.
def transform_corr_to_y(corr, ml, n_vars):
    ncorr = einops.rearrange(corr[:, :n_vars:], "b c1 (t c2) -> b c1 c2 t", t=ml)
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
