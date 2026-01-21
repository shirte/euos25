from ..training import NeuMFTraining
from .config import wandb_project

training = NeuMFTraining(
    model_name="model_2",
    mode="binary",
    use_absorbance=False,
    normalize_by_plate=False,
    use_plate_indices=True,
    use_plate_iterations=False,
    parameters=dict(
        embedding_dim=323,
        hidden_dim=75,
        num_layers=4,
        dropout=0.287,
        mlp_hidden_dim=21,
        mlp_num_layers=4,
        mlp_dropout=0.0359,
        learning_rate=0.0021,
        weight_decay=0.00003,
        plate_dropout=0,
        initial_plate_mean=0.0,
        initial_plate_std=0.0,
    ),
    wandb_project=wandb_project,
)

if __name__ == "__main__":
    training.eval_all_folds()
