# Minimum reproducible code

from pathlib import Path
from utils.causal_model import CausalModel # architecture module 
from utils.cp_utils import create_example_data # prediction module
from utils.plotting_utils import plot_summary_from_pred # plotting module


# Data generation and normalization
n = 1000
df = create_example_data(n)
variable_names = list(df.columns)

# Model loading and prediction
models_path = 'res'
model_name = 'lcm_CI_RH_12_3_merged_290k' 

model = CausalModel(model_name = model_name, model_path = Path(models_path) / f"{model_name}.ckpt") #Architecture_PL.load_from_checkpoint(Path(models_path) / f"{model_name}.ckpt")
pred = model.predict(df, max_lag_to_predict = 1)

# Results plot 
plot_summary_from_pred(pred, variable_names, plt_thr = 0.25)