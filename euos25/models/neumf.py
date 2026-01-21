import lightning as L
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from ..data import label_columns, num_plates


def build_mlp(input_dim, output_dim, hidden_dim, num_layers, dropout):
    layers = []
    dim = input_dim
    if num_layers <= 1:
        layers.append(nn.Linear(dim, output_dim))
        return nn.Sequential(*layers)

    for _ in range(num_layers - 1):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        dim = hidden_dim

    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class NeuMF(L.LightningModule):
    def __init__(
        self,
        # recommender
        embedding_dim=100,
        compound_encoder: nn.Module = None,
        mlp_hidden_dim=128,
        mlp_num_layers=2,
        mlp_dropout=0.2,
        # data
        num_tasks: int = len(label_columns),
        loss="bce",
        use_plate_bias: bool = True,
        use_plate_iterations: bool = False,
        initial_plate_mean: float = 0.0,
        initial_plate_std: float = 0.1,
        # optimization
        learning_rate: float = 1e-3,
        weight_decay: float = 0,
        plate_dropout: float = 0.5,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.use_plate_bias = use_plate_bias
        self.plate_dropout = plate_dropout

        if self.use_plate_bias:
            # plate baselines (plate x task)
            if use_plate_iterations:
                self.num_plates = num_plates * 4
            else:
                self.num_plates = num_plates
            self.plate_bias = nn.Parameter(torch.empty(self.num_plates, num_tasks))
            nn.init.normal_(
                self.plate_bias, mean=initial_plate_mean, std=initial_plate_std
            )

        # assay encoders
        self.assay_gmf = nn.Embedding(num_tasks, embedding_dim)
        self.assay_mlp = nn.Embedding(num_tasks, embedding_dim)

        # compound encoder
        self.compound_encoder = compound_encoder

        # MLP path
        self.mlp = build_mlp(
            input_dim=2 * embedding_dim,
            output_dim=embedding_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            dropout=mlp_dropout,
        )

        # final predictor combines GMF + MLP
        self.output = nn.Linear(2 * embedding_dim, 1)

        # configure loss
        self.bce_loss_fn = nn.functional.binary_cross_entropy_with_logits
        self.mse_loss_fn = nn.functional.mse_loss
        if loss == "bce":
            self.loss_fn = self.bce_loss_fn
        elif loss == "mse":
            self.loss_fn = self.mse_loss_fn

        self.save_hyperparameters(ignore=["compound_encoder"])

    def forward(self, batch):
        compound_representations = self.compound_encoder(batch)

        assay_gmf = self.assay_gmf(batch.task_id)
        assay_mlp = self.assay_mlp(batch.task_id)

        gmf_vector = assay_gmf * compound_representations
        mlp_input = torch.cat([assay_mlp, compound_representations], dim=1)
        mlp_vector = self.mlp(mlp_input)

        fused = torch.cat([gmf_vector, mlp_vector], dim=1)
        pre_activations = self.output(fused).view(-1)

        if self.use_plate_bias and self.training:
            plate_indices = batch.plate_id.view(-1)
            assert torch.all(
                (plate_indices >= 0) & (plate_indices < self.num_plates)
            ), "Invalid plate indices encountered during training"
            task_indices = batch.task_id.view(-1)
            plate_bias = self.plate_bias[plate_indices, task_indices]
            if self.plate_dropout > 0:
                # Drop plate biases by zeroing a random subset.
                mask = torch.rand_like(plate_bias) > self.plate_dropout
                plate_bias = plate_bias * mask
            return pre_activations + plate_bias

        return pre_activations

    def _select_loss_fn(self, batch):
        if hasattr(batch, "label_type"):
            label_type = int(batch.label_type.view(-1)[0].item())
            if label_type == 0:
                return self.bce_loss_fn
            if label_type == 1:
                return self.mse_loss_fn
        return self.loss_fn

    def _compute_loss(self, pre_activations, batch):
        loss_fn = self._select_loss_fn(batch)
        return loss_fn(pre_activations, batch.y.float())

    def training_step(self, batch, batch_idx):
        pre_activations = self.forward(batch)

        loss = self._compute_loss(pre_activations, batch)
        self.log("train_loss", loss)

        if self.use_plate_bias:
            # We want to see the magnitude of the bias being learned
            avg_bias_magnitude = self.plate_bias.abs().mean()
            self.log("train_avg_plate_bias", avg_bias_magnitude)

        return loss

    def on_validation_epoch_start(self):
        self._val_logits = []
        self._val_targets = []
        self._val_tasks = []

    def validation_step(self, batch, batch_idx):
        pre_activations = self.forward(batch)

        loss = self._compute_loss(pre_activations, batch)

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

        loss = self.bce_loss_fn(logits, targets.float())
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        pre_activations = self.forward(batch)

        loss = self._compute_loss(pre_activations, batch)
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
                threshold=1e-6,  # min change to count as improvement
                min_lr=1e-6,
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
