import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as opt

from utils.informer import transformer
from utils.tools import custom_corr_regularization, weighted_mse

# This class extends the pl.LightningModule and automates the loading the Causal-Pretraining models, as presented 
# in (Stein et al, 2024): https://github.com/Gideon-Stein/CausalPretraining, where the presented LCMs are based on.    
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
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.n_vars = n_vars
        self.max_lags = max_lags

        self.loss_type = loss_type
        self.val_metric = val_metric
        self.optimizer_lr = optimizer_lr
        self.regression_head = regression_head
        self.corr_input = corr_input
        self.weight_decay = weight_decay
        self.link_thresholds = link_thresholds
        self.corr_regularization = corr_regularization
        self.trans_max_ts_length = trans_max_ts_length
        self.mlp_max_ts_length = mlp_max_ts_length
        self.loss_term_scaling = torch.Tensor([2, 0.25, 0.25])
        self.full_representation_mode = full_representation_mode
        self.loss_scaling = {}
        self.distinguish_mode = distinguish_mode
        if self.distinguish_mode:
            self.regression_head = True

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
                regression_head=self.regression_head,
                corr_input=corr_input,
            )
        else:
            print("MODEL TYPE NOT KNOWN!")

        self.regression_loss = self.regression_loss_init()
        self.classifier_loss = self.classifier_loss_init()
        self.weights = torch.Tensor([1, 1, 1])


    def regression_loss_init(self):
        if self.distinguish_mode:
            return nn.BCEWithLogitsLoss()
        if self.regression_head:
            return nn.MSELoss() # when Regression head is used, the regression loss is MSE
        else:
            return None

    # Classifier loss initialization, either MSE, WMSE, MAE or binary cross-entropy
    def classifier_loss_init(self):
        if self.loss_type == "mse":
            print("init with MSE")
            return nn.MSELoss()
        if self.loss_type == "wmse":
            print("init with WMSE")
            return weighted_mse()
        if self.loss_type == "mae":
            print("init with mae")
            return nn.L1Loss()

        elif self.loss_type == "bce":
            return nn.BCEWithLogitsLoss()
        else:
            return None

    def non_train_step(self, batch, name="no_name"):
        inputs, labels = batch

        y_ = self.model(inputs)
        if self.distinguish_mode:
            reg_loss = self.regression_loss(y_[1], labels)
            class_loss = torch.zeros((1, 1), device=reg_loss.device) + 1e-10
            corr_loss = torch.zeros((1, 1), device=reg_loss.device) + 1e-10
            self.log(
                name + "_loss",
                reg_loss * self.loss_term_scaling[2], sync_dist=True,
                prog_bar=True,
            )

        else:
            if self.regression_head:
                y_1 = y_[0]
                l1 = labels[0]
                l2 = labels[1]
                y_2 = y_[1]
                reg_loss = self.regression_loss(y_2, l2)
                self.log(
                    name + "_reg_loss",
                    reg_loss * self.loss_term_scaling[2],
                    sync_dist=True,
                    prog_bar=True,
                )

            else:
                y_1 = y_ # y_1 consists of the output of the model
                l1 = labels # l1 consists of the labels for the classifier
                reg_loss = 0

            if self.corr_regularization:
                raw_data = inputs[0] if self.corr_input else inputs
                corr_loss = custom_corr_regularization(torch.sigmoid(y_1), raw_data)
                self.log(
                    name + "_corr_loss",
                    corr_loss * self.loss_term_scaling[1],
                    sync_dist=True,
                    prog_bar=True,
                )
            else:
                corr_loss = 0

            class_loss = self.classifier_loss(y_1, l1)
            self.log(
                name + "_class_loss",
                class_loss * self.loss_term_scaling[0],
                sync_dist=True,
                prog_bar=True,
            )

            loss = ( # \lambda_0 * class_loss + \lambda_1 & CR_loss + \lambda_2 & RH_loss
                class_loss * self.loss_term_scaling[0]
                + corr_loss * self.loss_term_scaling[1]
                + reg_loss * self.loss_term_scaling[2]
            )
            mse = self.mse(torch.sigmoid(y_1), l1) # computes the MSE between the sigmoid of the output and the labels

            self.calc_log_F1_metrics(y_1, l1, name=name)
            self.log(name + "_MSE", mse, sync_dist=True, prog_bar=True)
            self.log(name + "_output_mean", y_1.mean(), sync_dist=True, prog_bar=True)
            self.log(name + "_loss", loss, sync_dist=True, prog_bar=True)

    def test_step(self, batch, _):
        self.non_train_step(batch, name="test")

    def configure_optimizers(self):
        optim = opt.AdamW(
            self.model.parameters(),
            lr=self.optimizer_lr, 
            weight_decay=self.weight_decay,
        )
        schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.2)

        return [optim], [{"scheduler": schedule, "monitor": "train_loss"}]
