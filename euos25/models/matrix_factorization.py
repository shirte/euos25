import lightning as L
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from ..data import label_columns


class MatrixFactorization(L.LightningModule):
    def __init__(
        self,
        # recommender
        embedding_dim=100,
        compound_encoder: nn.Module = None,
        # data
        loss="bce",
        # optimization
        learning_rate: float = 1e-3,
        weight_decay: float = 0,  # 1e-5
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # assay encoder
        self.assay_representations = nn.Embedding(len(label_columns), embedding_dim)

        # assay bias
        self.assay_bias = nn.Embedding(len(label_columns), 1)
        nn.init.zeros_(self.assay_bias.weight)

        # compound encoder
        self.compound_encoder = compound_encoder

        # configure loss
        if loss == "bce":
            self.loss_fn = nn.functional.binary_cross_entropy_with_logits
        elif loss == "mse":
            self.loss_fn = nn.functional.mse_loss

        self.save_hyperparameters(ignore=["compound_encoder"])

    def forward(self, batch):
        assay_representations = self.assay_representations(batch.task_id)
        compound_representations = self.compound_encoder(batch)

        pre_activations = (assay_representations * compound_representations).sum(dim=1)

        return pre_activations

    def training_step(self, batch, batch_idx):
        pre_activations = self.forward(batch)

        loss = self.loss_fn(pre_activations, batch.y.float())
        self.log("train_loss", loss)

        return loss

    def on_validation_epoch_start(self):
        self._val_logits = []
        self._val_targets = []
        self._val_tasks = []

    def validation_step(self, batch, batch_idx):
        pre_activations = self.forward(batch)

        loss = self.loss_fn(pre_activations, batch.y.float())

        self._val_logits.append(pre_activations.detach().cpu())
        self._val_targets.append(batch.y.detach().cpu())
        self._val_tasks.append(batch.task_id.detach().cpu())

        return loss

    def on_validation_epoch_end(self):
        logits = torch.cat(self._val_logits)
        targets = torch.cat(self._val_targets)
        task_ids = torch.cat(self._val_tasks)

        aucs = {}
        for task_id, label in enumerate(label_columns):
            mask = task_ids == task_id
            logits_task = logits[mask]
            targets_task = targets[mask]
            aucs[label] = roc_auc_score(targets_task, logits_task)

        for label, auc in aucs.items():
            self.log(f"val_auc_{label}", auc)

        self.log("val_mean_auc_transmittance", (aucs["t340"] + aucs["t450"]) / 2)
        self.log("val_mean_auc_fluorescence", (aucs["f340"] + aucs["f480"]) / 2)
        self.log(
            "val_mean_auc",
            (aucs["t340"] + aucs["t450"] + aucs["f340"] + aucs["f480"]) / 4,
        )

        loss = self.loss_fn(logits, targets.float())
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        pre_activations = self.forward(batch)

        loss = self.loss_fn(pre_activations, batch.y.float())
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx):
        pre_activations = self.forward(batch)

        return torch.sigmoid(pre_activations)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,  # shrink LR
                patience=10,  # wait N val checks without improvement
                threshold=1e-8,  # min change to count as improvement
                min_lr=1e-6,
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
