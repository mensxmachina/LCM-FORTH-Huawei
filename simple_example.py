## minimum reproducible code
from pathlib import Path
from utils.model_wrapper import Architecture_PL # defines the architecture
from utils.cp_utils import set_seed, create_example_data, run_cp_and_parse_res # prediction module
from utils.plotting_utils import plot_summary_from_pred # plotting module


# data generation and normalization
set_seed(42)
n = 1000
df = create_example_data(n)
variable_names = list(df.columns)


# model loading and prediction
models_path = 'res'
model_name = 'lcm_CI_RH_12_3_merged_290k' 

model = Architecture_PL.load_from_checkpoint(Path(models_path) / f"{model_name}.ckpt")
pred = run_cp_and_parse_res(model_name, model = model, df = df, max_lag = 2, seed = 42)

# result plotting
plot_summary_from_pred(pred, variable_names, plt_thr = 0.75)
