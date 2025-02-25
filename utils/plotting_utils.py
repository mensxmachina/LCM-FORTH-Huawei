import matplotlib.pyplot as plt
import networkx as nx
import torch

def plot_summary_from_pred(adj_matrix: torch.Tensor, variable_names: list, plt_thr: float):
    """
    Plots a graph where nodes represent variables and edges represent connections over time
    based on a 3D adjacency tensor. plt_thr controls the graph sparsity: the higher plt_thr, the more sparse the graph

    Args:
        adj_matrix (torch.Tensor): 3-dimensional adjacency tensor of shape (n_var, n_var, n_time_steps)
        variable_names (list): List of variable names - only used for labeling
        plt_thr (float): Graph sparsity

    Returns:
        None 
    """
    n_var, _, n_time_steps = adj_matrix.shape
    adj_matrix = adj_matrix[:n_var, :n_var, -n_time_steps:]

    G = nx.DiGraph()
    G.add_nodes_from(range(n_var))

    for t in range(n_time_steps):
        for i in range(n_var):
            for j in range(n_var):
                if adj_matrix[i, j, t] > plt_thr and i != j:
                    # Add an edge with the time step as an attribute
                    if G.has_edge(j, i):
                        # If an edge already exists, append the time step to the list
                        G[j][i]['time_steps'].append(f't-{n_time_steps-t}')
                    else:
                        # Otherwise, create a new edge with the first time step
                        G.add_edge(j, i, time_steps=[f't-{n_time_steps-t}'])

    # Draw the graph
    pos = nx.spring_layout(G, seed=42)  # Layout for node positions
    plt.figure(figsize=(5, 5))
    
    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold', labels = {i : n for i,n in enumerate(variable_names)})

    # Draw edge labels (time steps)
    edge_labels = {}
    for (i, j, data) in G.edges(data=True):
        time_labels = ", ".join(map(str, data['time_steps']))
        edge_labels[(i, j)] = time_labels

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title('Summary Graph')
    plt.show()