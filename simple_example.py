# Minimum reproducible code

from pathlib import Path
from utils.model_wrapper import Architecture_PL # architecture module 
from utils.cp_utils import create_example_data, run_cp_and_parse_res # prediction module
from utils.plotting_utils import plot_summary_from_pred # plotting module


# Data generation and normalization
n = 1000
df = create_example_data(n)
variable_names = list(df.columns)

# Model loading and prediction
models_path = 'res'
model_name = 'lcm_CI_RH_12_3_merged_290k' 

model = Architecture_PL.load_from_checkpoint(Path(models_path) / f"{model_name}.ckpt")
pred = run_cp_and_parse_res(model_name, model = model, df = df, max_lag = 2)

# Results plot 
plot_summary_from_pred(pred, variable_names, plt_thr = 0.75)
