# class to load the LCM models without using lightning.pytorch.
# There are two type of modules:
# - Architecture_PL, that is derived from the work of Stein et al. which corresponds to FORTH initial trials. Its model structure is in utils/informer.py.
# - LCMModule, that points to modified models as per FORTH subsequent trials. Its model structure is in utils/lcm_model.py.

from typing import Tuple, Optional
import torch
import torch.nn as nn
import lightning.pytorch as pl 
from utils.lcm_model import LCM as model
from utils.informer import transformer

# This class extends the pl.LightningModule and automates the loading the Causal-Pretraining models, as presented 
# in (Stein et al, 2024): https://github.com/Gideon-Stein/CausalPretraining, where the presented LCMs are based on.    
# Edit: this is adapted for not using lightning.pytorch
class Architecture_PL(pl.LightningModule):
    def __init__(
        self,
        n_vars=3,
        max_lags=3,
        trans_max_ts_length=500,
        mlp_max_ts_length=500,
        model_type="transformer",
        corr_input=True,
        loss_type="ce",
        val_metric="ME",
        regression_head=False,
        link_thresholds=[0.25, 0.5, 0.75],
        corr_regularization=False,
        soft_adapt=False,
        distinguish_mode=False,
        full_representation_mode=False,
        optimizer_lr=1e-4,
        weight_decay=0.01,
        d_model=32,
        n_heads=2,
        num_encoder_layers=2,
        d_ff=128,
        dropout=0.05,
        distil=True,
        **kwargs
    ):
        super().__init__()

        if distinguish_mode:
            regression_head = True

        self.model_type = model_type

        if self.model_type == "transformer":
            self.model = transformer(
                n_vars=n_vars,
                d_model=d_model,
                max_lags=max_lags,
                n_heads=n_heads,
                num_encoder_layers=num_encoder_layers,
                d_ff=d_ff,
                dropout=dropout,
                distil=distil,
                max_length=trans_max_ts_length,
                # regression_head=self.regression_head,
                regression_head=regression_head,
                corr_input=corr_input,
            )
        else:
            print("MODEL TYPE NOT KNOWN!")

class LCMModule(pl.LightningModule):
    def __init__(
        self,
        n_vars: int = 12,
        max_lag: int = 3,
        max_seq_len: int = 500,
        d_model: int = 16,
        n_heads: int = 1,
        n_blocks: int = 2,
        d_ff: int = 32,
        dropout_rate: float = 0.05,
        attention_distilation: bool = True,
        training_aids: bool = False,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.95, 0.98),
        scheduler_factor: float = 0.1,
        loss_balancing: str = None,
        patch_len: int = 16,
        stride: int = 4,
        num_patches: Optional[int] = None,
        **kwargs
    ):
        super().__init__()  # <- this is needed

        # Model initialization
        self.model = model(
            n_vars=n_vars,
            max_lag=max_lag,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_blocks=n_blocks,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            attention_distilation=attention_distilation,
            training_aids=training_aids,
            is_patched=True,
            patch_len=patch_len,
            stride=stride,
            num_patches=num_patches
         )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
