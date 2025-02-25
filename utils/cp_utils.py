import torch
import numpy as np
import pandas as pd

from utils.tools import lagged_batch_corr


def right_shift(arr, shift_by=1): # function to create time lagged causal relationships
    arr = list(arr)
    shift_by = shift_by % len(arr)  
    return np.array(arr[-shift_by:] + arr[:-shift_by])


def create_example_data(n):
    '''Creates an easy synthetic example to show the input data structure. Each time series is in a different column'''
    A = np.random.normal(size=n)
    B = 0.5 * right_shift(A, shift_by = 1) + np.random.normal(size=n)  # B(t) = 0.5 * A(t-1) + noise
    C = 0.6 * right_shift(B, shift_by = 1) + np.random.normal(size=n)  # C(t) = 0.6 * B(t-1) + noise

    df = pd.DataFrame({'A': A, 'B': B, 'C': C})
    return df


def run_cp_and_parse_res(model_name, model, df, max_lag = 1, seed = 42):
    '''Run the causal prediction and parses the results to output a n_vars x n_vars x time_lags matrix'''
    assert max_lag <= 3 and len(df.columns) <= 12, 'Maximum 3 time steps and 12 variables'
    X_test = torch.tensor(df.values, device='cpu', dtype=torch.float32)
    pred = run_cp(model_name, model = model, data = X_test, MAX_VAR = 12, MAX_LAG = 3)
    pred = pred[0].detach().clone()

    n_vars = len(df.columns)
    pred = pred[:n_vars, :n_vars, -max_lag:]
    return pred


def run_cp(model_name, model, data, MAX_VAR=None, MAX_LAG=None, seed=None) -> torch.Tensor:
    """ 
    A function that runs a Causal Pretraining model on a specific dataset, performing all the necessary internal steps. 

    Args: 
        - model_name (str): the name of the model; used to identify the max number of variables and lags (necessary for now)
        - model (Architecture_PL): the Causal Pretraining model
        - data (torch.Tensor): the data on which the model will perform inference on; should be a lagged adjacency tensor
        - MAX_VAR (int): (optional) the maximum number of variables; used to bypass the automatic model identification; defaults to None 
        - MAX_LAG (int): (optional) the maximum number of lags; used to bypass the automatic model identification; defaults to None
        - seed (int): (optional) the seed of the pseudorandom number generator when sampling from the normal distribution while performing
          random noise variable padding; defaults to None. Since variability is occuring during variable padding, ensuring a consistent seed
          guarantees stability during the inference step. 

    Return:
        - The predicted causal graph as a lagged adjacency matrix, of shape (MAX_VAR, MAX_VAR, MAX_LAG)
    """
    # get the actual model and set it up for evaluation  
    M = model.model
    M = M.to("cpu")
    M = M.eval()

    # model hyper-parameters 
    if (MAX_VAR is None) or (MAX_LAG is None):
        if model_name=="provided-trf-5V":
            MAX_VAR = 5
            MAX_LAG = 3
        elif (("deep" in model_name) or ("lcm" in model_name)) and (("_12_3" in model_name) or ("_10_3" in model_name)):
            MAX_VAR = 12
            MAX_LAG = 3
        else:
            raise ValueError(f"Model name was not identified - MAX_VAR & MAX_LAG are uknown therefore the process can not proceed.")
    
    # Check if lags exceed MAX_LAG or dimensionality exceeds MAX_VAR 
    assert data.shape[1]<=MAX_VAR, f"Variable dimension ({data.shape[1]}) larger than model's maximum variables ({MAX_VAR})."

    # Padding
    if seed is not None:
        torch.use_deterministic_algorithms(True)
        generator = torch.manual_seed(seed)
    else:
        generator = None

    VAR_DIF = MAX_VAR - data.shape[1]
    if data.shape[1] != MAX_VAR:
        data = torch.concat(
            [data, torch.normal(0, 0.01, (data.shape[0], VAR_DIF), generator=generator)], axis=1
        )

    # Normalization
    data = (data - data.min()) / (data.max() - data.min())

    # Check dimensions and decide whether a batched approach is needed
    if (data.shape[0]>600):
        
        # Predictions' placeholder
        bs_preds = []

        # Break into batches
        batches = [data[600*icr: 600*(icr+1), :] for icr in range(data.shape[0]//600)]
        if 600*(data.shape[0]//600) < data.shape[0]:
            batches.append(data[600*(data.shape[0]//600):, :])

        # Predict
        if ("corr" in model_name) or ("_CI_" in model_name) or (model_name=="provided-trf-5V"):
            if ("_RH_" in model_name):
                bs_preds = [torch.sigmoid(M((bs.unsqueeze(0), lagged_batch_corr(bs.unsqueeze(0), 3)))[0]) for bs in batches]
            else:
                bs_preds = [torch.sigmoid(M((bs.unsqueeze(0), lagged_batch_corr(bs.unsqueeze(0), 3)))) for bs in batches]
        else:
            bs_preds = [torch.sigmoid(M(bs.unsqueeze(0))) for bs in batches]
        preds = torch.cat(bs_preds, dim=0)

        pred = preds.mean(0)
        pred = pred.unsqueeze(0)

    else:
        # Predict
        if ("corr" in model_name) or ("_CI_" in model_name) or (model_name=="provided-trf-5V"):    
            if ("_RH_" in model_name):
                pred = torch.sigmoid(M((data.unsqueeze(0), lagged_batch_corr(data.unsqueeze(0), 3)))[0])
            else:    
                pred = torch.sigmoid(M((data.unsqueeze(0), lagged_batch_corr(data.unsqueeze(0), 3))))
        else:
            pred = torch.sigmoid(M(data.unsqueeze(0)))
    
    return pred
