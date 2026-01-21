from ..models import GIN, NeuMF
from .training import Training


class NeuMFTraining(Training):
    def _get_model(self, config, checkpoint=None):
        # GIN as compound encoder
        compound_encoder = GIN(
            hidden_dim=config.hidden_dim,
            output_dim=config.embedding_dim,
            num_layers=config.num_layers,
            dropout_p=config.dropout,
        )

        if checkpoint is not None:
            model = NeuMF.load_from_checkpoint(
                checkpoint, compound_encoder=compound_encoder
            )
            return model

        if self.mode == "binary":
            num_tasks = 4
        elif self.mode == "mixed":
            num_tasks = 6

        model = NeuMF(
            # recommender
            embedding_dim=config.embedding_dim,
            compound_encoder=compound_encoder,
            mlp_hidden_dim=config.mlp_hidden_dim,
            mlp_num_layers=config.mlp_num_layers,
            mlp_dropout=config.mlp_dropout,
            # data
            num_tasks=num_tasks,
            use_plate_bias=self.use_plate_indices or self.use_plate_iterations,
            use_plate_iterations=self.use_plate_iterations,
            initial_plate_mean=config.initial_plate_mean,
            initial_plate_std=config.initial_plate_std,
            # optimization
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            plate_dropout=config.plate_dropout,
        )

        return model
