from datetime import datetime
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import random
from utils.model_wrapper import LCMModule, Architecture_PL


def lagged_batch_corr(points, max_lags):
    # calculates the autocovariance matrix with a batch dimension
    # lagged variables are concated in the same dimension.
    # inpuz (B, time, var)
    # roll to calculate lagged cov:
    B, N, D = points.size()

    # we roll the data and add it together to have the lagged versions in the table
    stack = torch.concat(
        [torch.roll(points, x, dims=1) for x in range(max_lags + 1)], dim=2
    )

    mean = stack.mean(dim=1).unsqueeze(1)
    std = stack.std(dim=1).unsqueeze(1) + 1e-6 # avoiding division by zero
    diffs = (stack - mean).reshape(B * N, D * (max_lags + 1))

    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(
        B, N, D * (max_lags + 1), D * (max_lags + 1)
    )

    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    # make correlation out of it by dividing by the product of the stds
    corr = bcov / (
        std.repeat(1, D * (max_lags + 1), 1).reshape(
            std.shape[0], D * (max_lags + 1), D * (max_lags + 1)
        )
        * std.permute((0, 2, 1))
    )
    # we can remove backwards in time links. (keep only the original values)
    return corr[:, :D, D:]  # (B, D, D)


class CausalModel():

    def __init__(self, model_name, models_folder = './res', device = 'cpu', model_path = ''):
        self.model_name = model_name
        self.model = CausalModel._load_model_from_name(self.model_name, models_folder, device = device, ckpt_path = model_path)
        self.model_max_lag, self.model_max_var = CausalModel._calculate_max_lag_and_var(self.model_name)
        self.device = device

    @staticmethod
    def _load_model_from_name(model_name, models_folder, device, ckpt_path = ''):
        if ckpt_path == '':
            ckpt_path = Path(models_folder) / f"{model_name}.ckpt" 

        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found at expected path: {ckpt_path}")

        try:
            return Architecture_PL.load_from_checkpoint(ckpt_path, map_location = device)
        except Exception as e1:
            try:
                return LCMModule.load_from_checkpoint(ckpt_path, map_location = device)
            except Exception as e2:
                if True:
                    raise RuntimeError(
                        f"Failed to load checkpoint '{ckpt_path}' with either loader.\n"
                        f"Architecture_PL error: {e1}\n"
                        f"LCMModule error: {e2}\n"
                        # f"InformerModule error: {e3}"
                    )

    @staticmethod
    def _calculate_max_lag_and_var(model_name): #to change with new models
        return 3, 12

    def predict(self, df : pd.DataFrame, max_lag_to_predict, verbose = 0, seed = 42, rotation_invariant_prediction = False, prior: torch.Tensor=None, belief: torch.tensor=None):

        columns = list(df.columns)
        res = self._get_prediction_matrix(df, columns, max_lag_to_predict, verbose = verbose, seed = seed)
        if not rotation_invariant_prediction:
            # permutation_columns = list(permutations(columns)) # too time consuming
            permutation_columns = [columns[i:] + columns[:i] for i in range(len(columns))]
            vals = []
            for p in permutation_columns:
                pred_matrix = self._get_prediction_matrix(df, list(p), max_lag_to_predict, verbose = verbose, seed = seed, prior = prior, belief = belief)
                pred_mean = np.mean(pred_matrix.reshape(-1))
                vals.append((pred_mean, p))

            best_mean, best_perm = max(vals, key=lambda x: x[0])
            pred_best = self._get_prediction_matrix(df, list(best_perm), max_lag_to_predict, verbose = verbose, seed = seed, prior = prior, belief = belief)
            col_order = [best_perm.index(col) for col in columns]
            res = pred_best[col_order, :, :]            # reorder rows
            res = res[:, col_order, :]                  # reorder columns
        return res 


    def _get_prediction_matrix(self, df, columns, max_lag_to_predict, verbose, seed, prior: torch.Tensor=None, belief: torch.tensor=None):
        assert max_lag_to_predict <= self.model_max_lag, 'Prediction exceeds model max lag limit'
        assert len(columns) <= self.model_max_var, 'Prediction exceeds model max variables number limit'
        
        # Covert time-series to tensor
        n_vars = len(columns)
        X_test = torch.tensor(df[columns].values, device=self.device, dtype=torch.float32)

        if verbose > 100:
            print(f'{datetime.now().time()} Running model: {self.model_name}')

        pred = run_informer(self.model_name, model = self.model, data = X_test, MAX_VAR=self.model_max_var, MAX_LAG= self.model_max_lag, seed = seed, prior = prior, belief = belief)

        # if isinstance(self.model, Architecture_PL):
        #     pred = run_cp(self.model_name, model = self.model, data = X_test, MAX_VAR=self.model_max_var, MAX_LAG= self.model_max_lag, seed = seed)
        # elif isinstance(self.model, LCMModule):
        #     pred = run_lcm(self.model_name, model = self.model, data = X_test, MAX_VAR=self.model_max_var, MAX_LAG= self.model_max_lag, seed = seed, prior = prior, belief = belief)
        # elif isinstance(self.model, InformerModule):
        #     pred = run_informer(self.model_name, model = self.model, data = X_test, MAX_VAR=self.model_max_var, MAX_LAG= self.model_max_lag, seed = seed, prior = prior, belief = belief)
        if self.model != 'cpu':
            pred = pred.cpu()
        pred = pred[0].detach().clone().numpy()
        
        if verbose > 100:
            print(f'{datetime.now().time()} End of inference')

        return pred[:n_vars, :n_vars, -max_lag_to_predict:] # cropping the predicted matrix to de desired dimensions

def seed_everything(seed: int = 42):
    random.seed(seed)                     # Python random
    np.random.seed(seed)                  # NumPy random
    torch.manual_seed(seed)               # PyTorch CPU
    torch.cuda.manual_seed(seed)          # PyTorch GPU
    torch.cuda.manual_seed_all(seed)      # All GPUs, if multiple
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_informer(model_name, model, data, MAX_VAR, MAX_LAG, seed, prior, belief):
    M = model.model
    # M = M.to("cpu")
    M = M.eval() 

    if (MAX_VAR is None) or (MAX_LAG is None):
        if model_name=="provided-trf-5V":
            MAX_VAR = 5
            MAX_LAG = 3
        elif ("deep" in model_name) and ("_10_3" in model_name) or ("deep" in model_name) and ("_12_3" in model_name) or ("lcm" in model_name) and ("_12_3" in model_name):
            MAX_VAR = 12
            MAX_LAG = 3
        else:
            raise ValueError(f"Model name was not identified - MAX_VAR & MAX_LAG are uknown therefore the process can not proceed.")

    assert data.shape[1]<=MAX_VAR, f"Variable dimension ({data.shape[1]}) larger than model's maximum variables ({MAX_VAR})."


    # Padding
    if seed is not None:
        # torch.use_deterministic_algorithms(True)c
        # generator = torch.manual_seed(seed)
        generator = torch.Generator(device=data.device)
        # pl.seed_everything(42, workers=True, verbose = False)
        seed_everything(42)
    else:
        generator = None

    X_cpd = data

    # Normalization
    X_cpd = (X_cpd - X_cpd.min()) / (X_cpd.max() - X_cpd.min())

    # Check dimensions to make sure they do not exceed the model's MAX_LAG & MAX_VAR
    assert X_cpd.shape[1]<=MAX_VAR, \
        f"ValueError: input time-series have {X_cpd.shape[1]} variables, while the current model supports at most {MAX_VAR}"

    # Padding
    VAR_DIF = MAX_VAR - X_cpd.shape[1]
    if X_cpd.shape[1] != MAX_VAR: # if the number of variables is less than the maximum
        X_cpd = torch.concat(
            [X_cpd, torch.normal(0, 0.01, (X_cpd.shape[0], VAR_DIF), generator=generator, device=X_cpd.device)], axis=1 # pad with noise
        )

    if (X_cpd.shape[0]>600):
    
        bs_preds = []
        batches = [X_cpd[600*icr: 600*(icr+1), :] for icr in range(X_cpd.shape[0]//600)]
        if 600*(X_cpd.shape[0]//600) < X_cpd.shape[0]:
            batches.append(X_cpd[600*(X_cpd.shape[0]//600):, :])

        if ("corr" in model_name) or ("_CI_" in model_name) or (model_name=="provided-trf-5V"):
            if ("_RH_" in model_name):
                with torch.no_grad():
                    bs_preds = [torch.sigmoid(M((bs.unsqueeze(0), lagged_batch_corr(bs.unsqueeze(0), 3)))[0]) for bs in batches]
            else:
                with torch.no_grad():
                    bs_preds = [torch.sigmoid(M((bs.unsqueeze(0), lagged_batch_corr(bs.unsqueeze(0), 3)))) for bs in batches]
        else:
            with torch.grad():
                bs_preds = [torch.sigmoid(M(bs.unsqueeze(0))) for bs in batches]
        preds = torch.cat(bs_preds, dim=0)

        pred = preds.mean(0)
        pred = pred.unsqueeze(0)

    else:
        if ("corr" in model_name) or ("CI" in model_name) or (model_name=="provided-trf-5V"):    
            if ("RH" in model_name):
                with torch.no_grad():
                    pred = torch.sigmoid(M((X_cpd.unsqueeze(0), lagged_batch_corr(X_cpd.unsqueeze(0), 3)))[0])
            else:    
                with torch.no_grad():
                    pred = torch.sigmoid(M((X_cpd.unsqueeze(0), lagged_batch_corr(X_cpd.unsqueeze(0), 3))))
                    #pred = torch.sigmoid(M((X_cpd.unsqueeze(0), lagged_batch_corr(X_cpd.unsqueeze(0), 3), lagged_batch_transfer_entropy(X_cpd.unsqueeze(0), MAX_LAG)))) 
        else:
            with torch.no_grad():
                pred = torch.sigmoid(M(X_cpd.unsqueeze(0)))

    return pred

