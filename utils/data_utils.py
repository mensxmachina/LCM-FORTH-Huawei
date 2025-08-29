import numpy as np
import pandas as pd


def right_shift(arr: np.ndarray, shift_by: int=1): # function to create time lagged causal relationships
    """
    Shifts a numpy array to the right by a specified number of positions.

    Args:
        arr (np.ndarray): The input array to be shifted.
        shift_by (int): The number of positions to shift the array to the right.

    Returns: 
        np.ndarray: The shifted array.
    """

    arr = list(arr)
    shift_by = shift_by % len(arr)  

    return np.array(arr[-shift_by:] + arr[:-shift_by])


def create_example_data(n: int):
    """
    Creates an easy synthetic example to show the input data structure. Each time series corresponds to a different column in the DataFrame.
    The illustrative example consists of 3 variables A, B, and C, with the causal relationships A -> B and B -> C.
    The temporal SCM is of the form $A_t \sim \mathcal{N}(0,1)$, $B_t := 0.5 * A_{t-1} + \mathcal{N}(0,1)$ and $C_t := 0.6 * B_{t-1} + \mathcal{N}(0,1)$.

    Args:
        n (int): The number of time steps to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the synthetic time series data.
    """

    A = np.random.normal(size=n)
    B = 0.5 * right_shift(A, shift_by = 1) + np.random.normal(size=n)  # B(t) = 0.5 * A(t-1) + noise
    C = 0.6 * right_shift(B, shift_by = 1) + np.random.normal(size=n)  # C(t) = 0.6 * B(t-1) + noise

    df = pd.DataFrame({'A': A, 'B': B, 'C': C})

    return df

