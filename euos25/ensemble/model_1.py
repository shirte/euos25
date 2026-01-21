from ..training import MatrixFactorizationTraining
from .config import wandb_project

training = MatrixFactorizationTraining(
    model_name="model_1",
    mode="binary",
    parameters=dict(
        embedding_dim=64,
        hidden_dim=64,
        num_layers=5,
        dropout=0.2,
        batch_size=7372,
        learning_rate=0.001,
        weight_decay=0,
    ),
    wandb_project=wandb_project,
)

if __name__ == "__main__":
    training.eval_all_folds()
