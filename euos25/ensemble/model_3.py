from ..training import NeuMFTraining
from .config import wandb_project

training = NeuMFTraining(
    model_name="model_3",
    mode="mixed",
    use_absorbance=True,
    normalize_by_plate=False,
    use_plate_indices=True,
    use_plate_iterations=False,
    parameters=dict(
        embedding_dim=64,
        hidden_dim=64,
        num_layers=5,
        dropout=0.2,
        mlp_hidden_dim=128,
        mlp_num_layers=2,
        mlp_dropout=0.2,
        learning_rate=0.001,
        weight_decay=0,
        plate_dropout=0.5,
        initial_plate_mean=0.0,
        initial_plate_std=0.0,
    ),
    wandb_project=wandb_project,
)

if __name__ == "__main__":
    training.eval_all_folds()
