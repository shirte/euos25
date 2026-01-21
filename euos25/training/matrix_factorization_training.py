from ..models import GIN, MatrixFactorization
from .training import Training


class MatrixFactorizationTraining(Training):
    def _get_model(self, config, checkpoint=None):
        # GIN as compound encoder
        compound_encoder = GIN(
            hidden_dim=config.hidden_dim,
            output_dim=config.embedding_dim,
            num_layers=config.num_layers,
            dropout_p=config.dropout,
        )

        if checkpoint is not None:
            model = MatrixFactorization.load_from_checkpoint(
                checkpoint, compound_encoder=compound_encoder
            )
            return model

        model = MatrixFactorization(
            # recommender
            embedding_dim=config.embedding_dim,
            compound_encoder=compound_encoder,
            # optimization
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        return model
