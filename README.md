# Large Causal Models on Time Series

This repository contains code and resources for testing Large Causal Models (LCMs). The aim is discovering temporal causal relationships in time-series datasets, using pretrained deep NN models. The LCM takes as input a temporal dataset $X \in \mathbb{R}^{N \times D}$ where $N$ is the sample size and $D$ the feature size (number of time-series). The output is a lagged adjacency tensor of shape $(N, N, \ell_\text{max})$ where $\ell_\text{max}$ is the hyperparemeter of the maximum assumed lag.
An high-level overview of this project is available on [our welcoming page](https://mensxmachina.github.io/LCM-FORTH-Huawei/).

## Installation

Install the dependencies from the `requirements.txt` file, either on your base environment or into an existing conda environment using

`pip install -r requirements.txt`

## Pre-trained weights

Obtain the LCM weights from the following URLs and place them into the `res` folder:
- `LCM_CI_CR_1.3M_12_3_joint_220k` (30 MB): [download link](https://figshare.com/articles/software/LCM_CI_CR_1_3M_12_3_joint_220k_ckpt/30022681) 
- `LCM_CI_9.6M_joint_220k_permuted_3` (127 MB): [download link](https://figshare.com/articles/software/LCM_CI_9_6M_joint_220k_permuted_3_ckpt/30022678) 
- `lcm_CI_RH_12_3_merged_290k` (4.6 GB): [download link](https://figshare.com/articles/software/lcm_CI_CR_12_3_merged_340k/30022684) 

---

## Illustrative Example


### 1. Import Required Libraries

At first, import the necessary modules for data generation, model prediction, and result visualization:


```python
from pathlib import Path
from utils.causal_model import CausalModel # architecture module 
from utils.data_utils import create_example_data # example data creation module
from utils.plotting_utils import plot_summary_from_pred, plot_summary_graph # plotting module
```

### 2. Generate Synthetic Data

We generate synthetic data with 1000 time samples, where each column represents a different time series. Data are Min-max normalized and random seed set to `42` for reproducibility.

```python
set_seed(42)

df = create_example_data(n=1000)
variable_names = list(df.columns)
```

### 3. Load the Pretrained Model

Load the `.ckpt` pretrained model for causal prediction:

```python
models_path = 'res'
model_name = 'lcm_CI_RH_12_3_merged_290k'

model = CausalModel(model_name = model_name, model_path = Path(models_path) / f"{model_name}.ckpt") 
```

### 4. Perform Causal Discovery

Run `model.predict` to perform causal discovery on the previous data. The `max_lag` parameter specifies the maximum time window size for analyzing causal relationships:

```python
# Run causal discovery with a maximum lag of 1
pred = model.predict(df, max_lag_to_predict = 1)
```

The result is a lagged adjacency tensor of shape `(N, N, max_lag)` where:

- `N` is the number of input time-series
- `pred[i, j, k]` represents the probability that the `j`-th time-series at time `t-(max_lag - k)` causes the `i`-th time-series at time `t`.

### 5. Visualize the Results

The predicted causal relationships can be visualized using `plot_summary_from_pred`. The `plt_thr` parameter controls the density of the graph: higher values result in fewer edges being displayed.

```python
plot_summary_from_pred(pred, variable_names, plt_thr=0.5)
```

In the resulting graph, an edge from time series A to B marked as `t-1` means that time series A at time `t-1` caused time series B at time `t`.

![Output plot of the summary graph.](media/summary.png)


### 6. Alternative Causal Discovery Method
As an alternative to using a specific causal model or threshold, the `get_best_graph` method can be applied. This method evaluates all available models and thresholds and returns the causal graph that optimally represents the relationships in the dataset.

```python
import utils.prediction_utils as pu
G = pu.get_best_graph(df, models_folder = models_path)
plot_summary_graph(G, variable_names)
```

The above example can be found in `simple_example.py`

---

### Limitations

We assume Causal Markov Condition and Faithfulness throughout. The following assumptions are also made:

- **Causal inference of up to 12 variables and 3 time lags**: The models can handle inputs up to 12 variables and $\ell_\text{max}=1,2,3$.

- **No contemporaneous effects**: The model discovers only the non-instantaneous relationships, i.e. the ones that occur with lag $\ell > 0$.

---

### Contacts

If you have questions, suggestions, or would like to collaborate, feel free to open an issue or reach out via mail at: wangmingxue1@huawei.com or bora.caglayan@huawei.com
