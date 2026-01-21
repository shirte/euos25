import glob
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
from loguru import logger
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from torch.utils.data import BatchSampler, ConcatDataset, RandomSampler
from torch_geometric.data import DataLoader

import wandb

from ..data import (
    LightningDatasetWrapper,
    RoundRobinBatchSampler,
    Tox25Dataset,
    label_columns,
)
from ..paths import MODEL_DIR


class Training(ABC):
    def __init__(
        self,
        model_name,
        mode,
        parameters,
        use_absorbance=False,
        use_plate_indices=False,
        use_plate_iterations=False,
        normalize_by_plate=False,
        normalize_labels=False,
        wandb_project=None,
    ):
        self.model_name = model_name
        self.mode = mode
        self.use_absorbance = use_absorbance
        self.use_plate_indices = use_plate_indices
        self.use_plate_iterations = use_plate_iterations
        self.normalize_by_plate = normalize_by_plate
        self.normalize_labels = normalize_labels
        self.parameters = parameters
        self.wandb_project = wandb_project
        self.dataset_kwargs = dict(
            use_absorbance=use_absorbance,
            use_plate_indices=use_plate_indices,
            use_plate_iterations=use_plate_iterations,
            normalize_by_plate=normalize_by_plate,
            normalize_labels=normalize_labels,
        )

    def checkpoint_dir(self, fold):
        return MODEL_DIR / self.model_name / str(fold)

    def train(self, fold):
        assert fold in {0, 1, 2, 3, 4}

        config = SimpleNamespace(**self.parameters)

        # set seed for reproducibility
        L.seed_everything(42)

        # prepare datasets
        train_set = f"full_train_fold_{fold}"
        val_set = f"full_test_fold_{fold}"

        if self.mode == "mixed":
            binary_dataset = Tox25Dataset(
                subset=train_set,
                mode="binary",
                **self.dataset_kwargs,
            )
            continuous_dataset = Tox25Dataset(
                subset=train_set,
                mode="continuous",
                **self.dataset_kwargs,
            )
            additional_datasets = []
            dataset_train = ConcatDataset(
                [binary_dataset, continuous_dataset] + additional_datasets
            )

            offsets = [0, len(binary_dataset)]
            for _dataset in additional_datasets:
                offsets.append(offsets[-1] + len(_dataset))

            def train_batch_sampler_generator(batch_size):
                samplers = [
                    BatchSampler(
                        RandomSampler(binary_dataset),
                        batch_size=batch_size,
                        drop_last=True,
                    ),
                    BatchSampler(
                        RandomSampler(continuous_dataset),
                        batch_size=batch_size,
                        drop_last=True,
                    ),
                ]
                for additional_dataset in additional_datasets:
                    samplers.append(
                        BatchSampler(
                            RandomSampler(additional_dataset),
                            batch_size=batch_size,
                            drop_last=True,
                        )
                    )
                return RoundRobinBatchSampler(samplers, offsets=offsets)
        else:
            dataset_train = Tox25Dataset(
                subset=train_set,
                mode=self.mode,
                **self.dataset_kwargs,
            )
            train_batch_sampler_generator = None

        # important: do not transform the validation set in any way
        dataset_val = Tox25Dataset(subset=val_set, mode="binary")

        batch_size = config.batch_size if hasattr(config, "batch_size") else 5000

        dm = LightningDatasetWrapper(
            dataset_train,
            dataset_val,
            batch_sampler_generator=train_batch_sampler_generator,
            batch_size=batch_size,  # initial batch size that will be tuned below
            num_workers=0,
        )

        # get model
        checkpoint_dir = self.checkpoint_dir(fold)
        checkpoint_files = glob.glob(str(checkpoint_dir / "*.ckpt"))
        checkpoint = next(iter(checkpoint_files), None)
        model = self._get_model(config, checkpoint=checkpoint)
        if checkpoint is not None:
            logger.info(f"Loaded model from checkpoint: {checkpoint}")
            return model

        if self.wandb_project is not None:
            wandb_logger = WandbLogger(project=self.wandb_project)
        else:
            wandb_logger = None

        trainer = L.Trainer(
            logger=wandb_logger,
            log_every_n_steps=1,
            max_epochs=1000,
            callbacks=[
                EarlyStopping(
                    monitor="val_mean_auc",
                    mode="max",
                    patience=30,
                    min_delta=1e-6,
                ),
                ModelCheckpoint(monitor="val_mean_auc", mode="max", save_top_k=1),
                LearningRateMonitor(),
            ],
            # disable logging
            enable_model_summary=False,
            enable_progress_bar=False,
        )

        # batch size tuning
        if not hasattr(config, "batch_size"):
            tuner = Tuner(trainer)
            tuner.scale_batch_size(
                model,
                datamodule=dm,
                mode="binsearch",
                init_val=5000,
                max_trials=3,
                # batch size varies for graph data sets -> leave some headroom
                margin=0.1,
            )

        # training
        if self.wandb_project is not None:
            with wandb.init(project=self.wandb_project, config=config) as run:
                run.tags += (f"fold_{fold}", self.mode)
                trainer.fit(model=model, datamodule=dm)
        else:
            trainer.fit(model=model, datamodule=dm)

        if self.wandb_project is not None:
            wandb_logger.experiment.finish()

        # copy checkpoint to model dir
        checkpoint_path = Path(trainer.checkpoint_callback.best_model_path)

        # keep file name
        checkpoint_name = checkpoint_path.name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            checkpoint_path,
            checkpoint_dir / checkpoint_name,
        )

        return model

    def eval_all_folds(self):
        for fold in range(5):
            self.eval(fold=fold, subset="val")
            self.eval(fold=fold, subset="submission")

    def eval(self, fold, subset="submission"):
        # this will train the model if not already trained
        # otherwise, it will load the model from checkpoint
        model = self.train(fold)

        # set seed for reproducibility
        L.seed_everything(42)

        #
        # evaluate on validation set
        #

        # important: do not transform the validation/test set in any way
        if subset == "val":
            dataset = Tox25Dataset(subset=f"full_test_fold_{fold}", mode="binary")
        elif subset == "submission":
            dataset = Tox25Dataset(subset="submission", mode="binary")

        trainer = L.Trainer(
            max_epochs=1000,
        )

        predict_loader = DataLoader(
            dataset,
            batch_size=10_000,
            shuffle=False,  # important for predict
            num_workers=0,
        )

        mol_ids = torch.cat([data.mol_id for data in dataset], dim=0).cpu().numpy()
        tasks = torch.cat([data.task_id for data in dataset], dim=0).cpu().numpy()
        y_pred = np.concatenate(trainer.predict(model, predict_loader))
        if subset == "val":
            y_true = torch.cat([data.y for data in dataset], dim=0).cpu().numpy()
        else:
            y_true = np.array([np.nan] * len(y_pred))

        # create dataframe
        df_predictions = pd.DataFrame(
            {
                "mol_id": mol_ids,
                "task_id": tasks,
                "y_true": y_true,
                "y_pred": y_pred.flatten(),
            }
        )

        # filter tasks
        df_predictions = df_predictions[df_predictions["task_id"].isin(range(4))]

        # map tasks
        task_id_mapping = {i: label for i, label in enumerate(label_columns)}
        df_predictions["task"] = df_predictions["task_id"].map(task_id_mapping)

        # compute performance metrics (AUC, MCC) on test fold
        if subset == "val":

            def _mcc(y_true, y_pred):
                # find best threshold
                best_mcc = -1
                for threshold in np.linspace(0.0, 1.0, num=101):
                    y_pred_bin = (y_pred >= threshold).astype(int)
                    mcc = matthews_corrcoef(y_true, y_pred_bin)

                    if mcc > best_mcc:
                        best_mcc = mcc

                return best_mcc

            def _metrics(df):
                auc = roc_auc_score(df["y_true"], df["y_pred"])
                mcc = _mcc(df["y_true"].values, df["y_pred"].values)

                return pd.Series({"auc": auc, "mcc": mcc})

            df_predictions.groupby("task").apply(_metrics).reset_index().to_csv(
                self.checkpoint_dir(fold) / "eval_metrics.csv", index=False
            )

        # create submission file
        df_submission = (
            df_predictions.pivot(index="mol_id", columns="task", values="y_pred")
            .reset_index()
            .sort_values("mol_id")
        )
        df_submission = df_submission[["t340", "t450", "f340", "f480"]]
        df_submission.columns = [
            "Transmittance(340)",
            "Transmittance(450)",
            "Fluorescence(340/450)",
            "Fluorescence(>480)",
        ]
        df_submission.to_csv(self.checkpoint_dir(fold) / "submission.csv", index=False)

    @abstractmethod
    def _get_model(self, config):
        pass
