from ..training import NeuMFTraining
from .config import wandb_project

training = NeuMFTraining(
    model_name="model_5",
    mode="binary",
    use_absorbance=False,
    normalize_by_plate=False,
    use_plate_indices=True,
    use_plate_iterations=False,
    parameters=dict(
        embedding_dim=66,
        hidden_dim=65,
        num_layers=22,
        dropout=0.0537,
        mlp_hidden_dim=445,
        mlp_num_layers=6,
        mlp_dropout=0.1813,
        learning_rate=0.00014,
        weight_decay=0.00033,
        plate_dropout=0,
        initial_plate_mean=0.0,
        initial_plate_std=0.0,
    ),
    wandb_project=wandb_project,
)

if __name__ == "__main__":
    training.eval_all_folds()
