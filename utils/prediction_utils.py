import pandas as pd
import numpy as np
import os
from pathlib import Path
from itertools import combinations
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.exceptions import ConvergenceWarning
import warnings
import networkx as nx
from joblib import Parallel, delayed
from utils.causal_model import CausalModel

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

def create_graph(
        adj_matrix : np.ndarray, 
        plt_thr : float, 
    ) -> nx.DiGraph: 
    """
    Create a directed summary graph from the adjacency matrix. The adjacency matrix is expected to be of shape (n_vars, n_vars, n_lags).

    Args:
        adj_matrix (np.ndarray): The adjacency matrix of shape (n_vars, n_vars, n_lags)
        plt_thr (float): The threshold for edge selection. All edges with lower probability will be removed.

    Returns:
        nx.DiGraph: The directed graph with edges based on the adjacency matrix.
    """

    n_vars, _, n_time_steps = adj_matrix.shape
    G = nx.DiGraph()
    G.add_nodes_from(range(n_vars))

    # Find all edges above threshold. adj[i,j, t] > plt_thr means that there is an edge from j to i at time t.
    # i_idx, j_idx, t_idx = np.where(adj_matrix > plt_thr)

    # for i, j in zip(i_idx, j_idx):
    #     if i != j and not G.has_edge(j, i):
    #         G.add_edge(j, i)
    
    for t in range(n_time_steps):
        for i in range(n_vars):
            for j in range(n_vars):
                if adj_matrix[i, j, t] > plt_thr and i != j:
                    # Add an edge with the time step as an attribute
                    if G.has_edge(j, i):
                        # If an edge already exists, append the time step to the list
                        G[j][i]['time_steps'].append(f't-{n_time_steps-t}')
                    else:
                        # Otherwise, create a new edge with the first time step
                        G.add_edge(j, i, time_steps=[f't-{n_time_steps-t}'])
    return G

def create_augmented_vars(
        df : pd.DataFrame
    ) -> pd.DataFrame:
    """
    Creates an augmented Datagrame with additional features combinations based on the original features. Currently it adds:
    - difference between consecutive values
    - inverse of the values
    - absolute values
    - product and rations of all the features

    Args:
        df (pd.DataFrame): the original dataframe to augment.

    Returns:
        pd.DataFrame: the augmented datagrame.
    """
    dfc = df.copy()
    cols = df.columns

    df_diff = df.diff().fillna(0)
    df_diff.columns = [f'diff({c})' for c in cols]
    
    df_inv = df.replace(0, np.nan).rpow(1)  # 1 / x
    df_inv.columns = [f'1/({c})' for c in cols]

    df_abs = df.abs()
    df_abs.columns = [f'|({c})|' for c in cols]

    prod_cols = {}
    div_cols = {}
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            prod_cols[f'({c1})*({c2})'] = df[c1] * df[c2]
            div_cols[f'({c1})/({c2})'] = df[c1] / (df[c2].replace(0, np.nan)+1e-5)

    dfc = pd.concat([df,
                     df_diff,
                     df_inv,
                     df_abs,
                     pd.DataFrame(prod_cols),
                     pd.DataFrame(div_cols)], axis=1)

    dfc.replace([np.inf, -np.inf], 0, inplace=True)
    dfc.fillna(0, inplace=True)
    return dfc

def get_deterministic_vars(
        coeffs : list, 
        coeffs_names : list, 
        tol_coeffs : float = 1e-2
    ) -> list:
    """
    Parses the coefficients names for the non-zero coefficients to find the deterministic variables they were calculated from.

    Args:
        coeffs (list) : list of coefficients from a lasso model
        coeffs_names (list) : list of names of the coefficients
        tol_coeffs (float, optional): tolerance to consider a coefficient as non-zero. Defaults to 1e-2.

    Returns:
        list: list of variable names that are deterministic with the target variable.
    """
    res = []
    sel = [c[0] for c in zip(coeffs_names, coeffs) if np.abs(c[1]) > tol_coeffs]
    for c1 in coeffs_names:
        if any([f'({c1})' in e for e in sel]):
            res.append(c1)
    return res

def process_combination(
        target_column, 
        var1, 
        var2, 
        df, 
        tol_deterministic, 
        tol_coeffs
    ) -> list:
    """
    Analyze a pair of variables to identify deterministic relationships with the target variable.
    Uses Lasso regression on augmented variables to detect linear dependencies.

    Args:
        target_column (str): Name of the target column in the dataframe.
        var1 (str): Name of the first independent variable.
        var2 (str): Name of the second independent variable.
        df (pd.DataFrame): Input dataframe containing all relevant columns.
        tol_deterministic (float): Threshold for R^2 to consider the relationship deterministic (e.g., 0.01 means that the two variables are determinstic if R^2 > 0.99).
        tol_coeffs (float): Threshold for coefficients to consider a variable deterministic (e.g., 0.01 means that all the variables with only coefficients lower than 0.01 will not considered as deterministic with the target).

    Returns:
        list of tuples: Each tuple is (target_column, deterministic_variable) if a deterministic relationship is detected.
                        Returns an empty list if no deterministic variables are found.
    """


    df_pair = df[[var1, var2]]
    df_aug = create_augmented_vars(df_pair)

    df_aug['target'] = df[target_column]
    df_aug.fillna(0, inplace=True)
    df_aug.replace([np.inf, -np.inf], 0, inplace=True)
    X = df_aug.drop(columns='target').values
    y = df_aug['target'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LassoCV()
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    score = r2_score(y, y_pred)

    if score > 1 - tol_deterministic: 
        deterministic_vars = get_deterministic_vars(
            coeffs_names = df_aug.drop(columns='target').columns.tolist(),
            coeffs = model.coef_.tolist(),
            tol_coeffs = tol_coeffs
        )

        return [(target_column, d) for d in deterministic_vars]

    return []

def discover_mat_combinations(
        df : pd.DataFrame, 
        tol_deterministic : float = 1e-2, 
        tol_coeffs : float = 5e-3, 
        n_jobs : int = -1, 
        thr_very_high_corr : float = 0.95
    ) -> list:
    """
    Discover deterministic or highly correlated variable combinations in a dataframe.

    This function:
        1. Identifies pairs of variables that are highly correlated.
        2. Uses Lasso regression (via `process_combination`) to find deterministic relationships between all combinations of three variables (Runs these combinations in parallel for efficiency).
        3. Returns both the highly correlated pairs and the deterministic relationships found.

    Args:
        df (pd.DataFrame): Input dataframe containing all relevant variables.
        tol_deterministic (float, optional): Threshold for R^2 to consider a combination deterministic. Defaults to 1e-2.
        tol_coeffs (float, optional): Threshold for Lasso coefficients to detect deterministic features. Defaults to 5e-3.
        n_jobs (int, optional): Number of parallel jobs for computation (-1 = use all cores). Defaults to -1.
        thr_very_high_corr (float, optional): Correlation threshold above which pairs are considered very correlated. Defaults to 0.95.

    Returns:
        list of tuples: List of deterministic variable relationships (target, deterministic_variable).
    """
    # step 1: find highly correlated pairs
    high_corr = []
    cor = df.corr()
    for i, c1 in enumerate(df.columns):
        for j,c2 in enumerate(df.columns):
            if j> i:
                if cor[c1][c2] > thr_very_high_corr:
                    high_corr.append((c1, c2))

    # step 2: find deterministic relationships
    # 2.1 create the combinations of features
    tasks = []
    for c in df.columns:
        features = [cc for cc in df.columns if cc != c]
        for var1, var2 in combinations(features, 2):
            if not (
                (var1, var2) in high_corr or 
                (var2, var1) in high_corr or 
                (c, var1) in high_corr or 
                (c, var2) in high_corr or 
                (var1, c) in high_corr or 
                (var2, c) in high_corr
            ): # avoiding already found high correlations
                tasks.append((c, var1, var2))

    # 2.2 Run all (c, var1, var2) combos in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_combination)(target, var1, var2, df, tol_deterministic, tol_coeffs)
        for target, var1, var2 in tasks
    )

    det = high_corr + [edge for sublist in results for edge in sublist]

    return det

def get_custom_correlation(
        df, 
        v1 : str, 
        v2 : str, 
        sample_fifths : bool = True, 
        include_lagged : bool = True, 
        symmetric : bool = True
    ) -> float:
    """
    Compute a custom correlation measure between two variables in a dataframe.

    This function optionally considers lagged correlation and can compute a symmetric correlation by taking the maximum of corr(v1, v2) and corr(v2, v1).

    Args:
        df (pd.DataFrame): Input dataframe containing the variables.
        v1 (str): Name of the first variable.
        v2 (str): Name of the second variable.
        sample_fifths (bool, optional): Whether to sample the data in fifths when computing correlation. Defaults to True.
        include_lagged (bool, optional): Whether to include lagged versions of the variables in the correlation computation. Defaults to True.
        symmetric (bool, optional): If True, returns the maximum of corr(v1, v2) and corr(v2, v1). Defaults to True.

    Returns:
        float: The custom correlation value between v1 and v2.
    """

    if not symmetric:
        return _get_custom_correlation(df, v1, v2, sample_fifths = sample_fifths, include_lagged = include_lagged)
    else:
        return max(_get_custom_correlation(df, v1, v2, sample_fifths = sample_fifths, include_lagged = include_lagged),
                   _get_custom_correlation(df, v2, v1, sample_fifths = sample_fifths, include_lagged = include_lagged))

def _get_custom_correlation(
        df, 
        v1 : str, 
        v2 : str, 
        sample_fifths : bool = True, 
        include_lagged : bool = True
    ) -> float:
    """
    Compute a custom correlation measure between two variables in a dataframe.
    This includes optional segment-based correlations (of the different fifths of dataset, if sample fifths is True) and lagged correlations.

    Args:
        df (pd.DataFrame): Input dataframe containing the variables.
        v1 (str): Name of the first variable.
        v2 (str): Name of the second variable.
        sample_fifths (bool, optional): Whether to split the data into 5 segments and compute correlations for each. Defaults to True.
        include_lagged (bool, optional): Whether to include lagged correlation (v1[t] vs v2[t+1]). Defaults to True.

    Returns:
        float: Maximum absolute correlation from all computed correlations.
    """

    segment_size = len(df) // 5

    corrs_thirds = [np.abs(np.corrcoef(
                        df[v1].values[i*int(segment_size):(i+1)*int(segment_size)], 
                        df[v2].values[i*int(segment_size):(i+1)*int(segment_size)]
                    )[0,1]) 
                    for i in range(5)] if sample_fifths else []
    
    corr_lagged = [np.abs(np.corrcoef(
            df[v1].iloc[:-1], df[v2].iloc[1:])[0,1]
        )] if include_lagged else []
    
    corrs_thirds_lagged = [np.abs(np.corrcoef(
        df[v1].values[i*int(segment_size):(i+1)*int(segment_size)-1], 
        df[v2].values[i*int(segment_size)+1:(i+1)*int(segment_size)])[0,1]
        ) for i in range(5)] if sample_fifths and include_lagged else []
    
    return np.max([np.abs(np.corrcoef(df[v1], df[v2])[0,1])] + corrs_thirds + corr_lagged + corrs_thirds_lagged)
    
def get_graph_score(        
        df: pd.DataFrame,
        G: nx.DiGraph,
        target: str = None,
        thr_good: float = 0.9,
        thr_OK: float = 0.5,
        prior_kn: list = None,
        sample_fifths: bool = False,
        include_lagged: bool = True,
        get_decomposed_score: bool = False
) -> float:
    """
    Score a directed graph based on its edges correlations and prior knowledge.

    The score considers:
    - How well discovered edges match high correlations in the data.
    - Agreement with prior known edges.
    - Graph density and sparsity penalties.
    - Optional focus on a target node.

    Args:
        df (pd.DataFrame): Dataframe with variables as columns.
        G (nx.DiGraph): Directed graph to score.
        target (str, optional): Node to prioritize in scoring, like a taget KPI to analyze. Defaults to None.
        thr_good (float, optional): Correlation threshold for "good" edges. Defaults to 0.9.
        thr_OK (float, optional): Correlation threshold for "OK" edges. Defaults to 0.5.
        prior_kn (list, optional): List of prior known edges (tuples). Defaults to None.
        sample_fifths (bool, optional): Whether to compute correlations in fifths of data. Defaults to False.
        include_lagged (bool, optional): Whether to include lagged correlations. Defaults to True.
        get_decomposed_score (bool, optional): If True, return intermediate scores for analysis. Defaults to False.

    Returns:
        float: Combined graph score 
        tuple (optional): decomposed score components (tuple) if get_decomposed_score=True.
    """
    corrs = {e: _get_custom_correlation(df, e[0], e[1], sample_fifths = sample_fifths, include_lagged = include_lagged) for e in G.copy().to_undirected().edges()}

    n_vars = len(G.nodes())
    n_edges = len(G.copy().to_undirected().edges)
    to_target = 0 if target is None else len([e for e in nx.ancestors(G, target)])
    target_penalty = 1 if target is None else min(to_target / (n_vars-2 + 1e-5), 1)


    n_prior, n_good, n_OK, n_bad = 0, 0, 0, 0

    score_visual = 0 # caps to 1, can be negative. 1 = perfect discovered edges, aka same as prior or good correlation
    if n_edges > 0: # score visual computation
        for e in G.to_undirected().edges():
            if e in prior_kn or (e[1],e[0]) in prior_kn:
                n_prior += 1
            else:
                if e not in corrs:
                    e = (e[1], e[0])
                if corrs[e] > thr_good:
                    n_good += 1
                elif corrs[e] > thr_OK:
                    n_OK += 1
                else:
                    n_bad += 1
        score_visual = (1.5 * n_prior + n_good - n_bad) / (n_edges - n_prior + 1.5 * len(prior_kn))
        assert score_visual <= 1

    #edge density score computation
    max_out_deg_und = max(dict(G.copy().to_undirected().degree()).values())
    dense_penalty = max(0, np.abs(n_edges - n_vars) - int(n_vars/4))# it's ok not to connect less than 1/4 of variables
    sparse_penalty = max(0, max_out_deg_und - 4) # max number of connections per node = 4
    score_edges =  1 - 1/n_vars * dense_penalty - .2 * sparse_penalty # caps at 1, can be negative. Is 1 if there are as many edges as variables

    if get_decomposed_score:
        return n_edges, n_prior, n_good, n_OK, n_bad, score_visual, score_edges, target_penalty, (.33*score_visual + .66*score_edges) * target_penalty
    return np.round((.33*score_visual + .66*score_edges) * target_penalty, 3)

def get_best_graph(
    df, 
    models_folder : str = 'res', 
    max_lag_to_predict : int = 1,
    target : str = None,
    device : str = 'cpu',
) -> tuple[nx.DiGraph, float]:
    '''
    Automatically discovers the best graph from the given dataframe and models list.

    Args:
        df (pd.DataFrame): Dataframe with variables as columns.
        max_lag_to_predict (int, optional): Maximum lag to consider in the prediction. Defaults to 1.
        target (str, optional): Target variable to focus on. Defaults to None.
        device (str, optional): Device to use for model inference ('cpu' or 'cuda:x'). Defaults to 'cpu'.
    
    Returns:
        nx.DiGraph: best causal graph
    '''
    models_paths_dict = {
        f: os.path.join(root, f) 
              for root, _, files in os.walk(models_folder) 
              for f in files if f.endswith('.ckpt')}
    prior = discover_mat_combinations(df)

    best_res = (None, None, '', None, -np.inf)
    res = []    
    # for model_name in models_paths_dict.keys():
    for model_name in [
        'LCM_CI_CR_1.3M_12_3_joint_220k.ckpt',
        'LCM_CI_9.6M_joint_220k_permuted_3.ckpt',
        'lcm_CI_RH_12_3_merged_290k.ckpt',
    ]:
        m = CausalModel(model_name, 
            models_folder = models_paths_dict[model_name].split(f'{model_name}.')[0], 
            device=device,
            model_path=models_paths_dict[model_name]
        )
        
        adj_matrix = m.predict(df, max_lag_to_predict = max_lag_to_predict)
        
        for plt_thr in [0.05*(i+1) for i in range(19)]:
            gg = create_graph(adj_matrix, plt_thr)
            
            gg = nx.relabel_nodes(gg, {i : v for i,v in enumerate(df.columns)})
            score = get_graph_score(df, gg, target = target, prior_kn = prior)
            gg = nx.relabel_nodes(gg, {v : i for i,v in enumerate(df.columns)})

            if score > best_res[4]:
                best_res = (model_name, plt_thr, 'joint_app_number', gg, score)
            res.append((model_name, plt_thr, 'joint_app_number', gg, score))

            if len(gg.edges()) == 0:
                break

    return best_res[3]