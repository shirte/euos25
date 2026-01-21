import glob
import os
from pathlib import Path
from typing import Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from torch_geometric.data import DataLoader

from ..data import (
    Tox25Dataset,
    label_columns,
)
from ..paths import PROJECT_ROOT
from .model_1 import training as training_model_1
from .model_2 import training as training_model_2
from .model_3 import training as training_model_3
from .model_4 import training as training_model_4
from .model_5 import training as training_model_5

training_objects = [
    training_model_1,
    training_model_2,
    training_model_3,
    training_model_4,
    training_model_5,
]

model_ids = list(range(1, len(training_objects) + 1))
folds = list(range(5))

extra_model_ids = {}
for label in label_columns:
    extra_model_ids[label] = [
        os.path.basename(path)
        for path in glob.glob(str(PROJECT_ROOT / "models" / "single" / label / "*"))
    ]


def model_dir(model_id: int, fold: Optional[int] = None) -> Path:
    model_name = f"model_{model_id}"
    if fold is None:
        return PROJECT_ROOT / "models" / model_name
    return PROJECT_ROOT / "models" / model_name / str(fold)


def load_eval_metrics(model_id: int, fold: int) -> pd.DataFrame:
    model_name = f"model_{model_id}"
    return pd.read_csv(str(model_dir(model_id, fold) / "eval_metrics.csv")).assign(
        fold=fold, model_name=model_name
    )


def predict(model_id: int, fold: int, subset="val", ranks=False) -> pd.DataFrame:
    if subset == "val":
        predictions_oof_path = model_dir(model_id, fold) / "predictions_oof.csv"
    elif subset == "submission":
        predictions_oof_path = model_dir(model_id, fold) / "predictions_submission.csv"

    if predictions_oof_path.exists():
        df_predictions = pd.read_csv(str(predictions_oof_path))
        if ranks:
            df_predictions = df_predictions.assign(
                y_pred=lambda df: df.groupby("task").y_pred.transform(
                    lambda x: x.rank(method="average", pct=True)
                )
            )
        return df_predictions

    L.seed_everything(42)

    if subset == "val":
        dataset = Tox25Dataset(subset=f"full_test_fold_{fold}", mode="binary")
    elif subset == "submission":
        dataset = Tox25Dataset(subset="submission", mode="binary")
    else:
        raise ValueError(f"Unknown subset: {subset}")

    trainer = L.Trainer(
        max_epochs=1000,
    )

    predict_loader = DataLoader(
        dataset,
        batch_size=10_000,
        shuffle=False,  # important for predict
        num_workers=0,
    )

    model = training_objects[model_id - 1].train(fold)

    mol_ids = torch.cat([data.mol_id for data in dataset], dim=0).cpu().numpy()
    tasks = torch.cat([data.task_id for data in dataset], dim=0).cpu().numpy()
    y_pred = np.concatenate(trainer.predict(model, predict_loader))
    if subset == "val":
        y_true = torch.cat([data.y for data in dataset], dim=0).cpu().numpy()
    else:
        y_true = np.array([np.nan] * len(y_pred))

    df_predictions = pd.DataFrame(
        {
            "mol_id": mol_ids,
            "task_id": tasks,
            "y_true": y_true,
            "y_pred": y_pred.flatten(),
        }
    )

    # map tasks
    task_id_mapping = {i: label for i, label in enumerate(label_columns)}
    df_predictions["task"] = df_predictions["task_id"].map(task_id_mapping)

    df_predictions.to_csv(str(predictions_oof_path), index=False)

    if ranks:
        df_predictions = df_predictions.assign(
            y_pred=lambda df: df.groupby("task").y_pred.transform(
                lambda x: x.rank(method="average", pct=True)
            )
        )

    return df_predictions


def eval(df_predictions) -> pd.DataFrame:
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

    return (
        df_predictions.groupby("task")
        .apply(_metrics, include_groups=False)
        .reset_index()
    )


def predictions_to_submission(df_predictions: pd.DataFrame) -> pd.DataFrame:
    # map tasks
    if "task" not in df_predictions.columns:
        df_predictions = df_predictions.copy()
        task_id_mapping = {i: label for i, label in enumerate(label_columns)}
        df_predictions["task"] = df_predictions["task_id"].map(task_id_mapping)

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

    # normalize each column so all values are between 0 and 1
    for col in df_submission.columns:
        min_val = df_submission[col].min()
        max_val = df_submission[col].max()
        df_submission[col] = (df_submission[col] - min_val) / (max_val - min_val)

    return df_submission


def eval_model(model_id) -> pd.DataFrame:
    if isinstance(model_id, int):
        df_predictions = pd.concat(
            [predict(model_id=model_id, fold=fold, ranks=True) for fold in folds]
        )
    elif isinstance(model_id, tuple) and len(model_id) == 2:
        task, extra_model_name = model_id

        # load predictions
        df_predictions = (
            pd.read_csv(
                PROJECT_ROOT
                / "models"
                / "single"
                / task
                / extra_model_name
                / f"{extra_model_name}_kfolds_eval_preds.csv"
            )
            .rename(columns={"y_proba": "y_pred", "id": "mol_id", "y_class": "y_true"})
            .assign(
                task=task,
                y_pred=lambda df: df.groupby("fold").y_pred.transform(
                    lambda x: x.rank(method="average", pct=True)
                ),
            )
            .drop(columns=["fold", "model"])
        )

    return eval(df_predictions)


def ensemble_ranks(df_predictions: pd.DataFrame, task: str) -> list:
    df_task = df_predictions[df_predictions["task"] == task].copy()
    df_task.sort_values("mol_id", inplace=True)
    ranks = df_task["y_pred"].rank(method="average", pct=True).values.tolist()
    return ranks


if __name__ == "__main__":
    # generate oof metrics per model
    df_oof_metrics = pd.concat(
        [eval_model(model_id).assign(model_id=model_id) for model_id in model_ids]
    )

    # compute model weights
    alpha = 20
    weight_per_model_and_task = []
    for task in label_columns:
        max_auc = df_oof_metrics[df_oof_metrics.task == task]["auc"].max()
        sum_exp_auc = (
            df_oof_metrics[df_oof_metrics.task == task]["auc"]
            .apply(lambda x: np.exp(alpha * (x - max_auc)))
            .sum()
        )
        for model_id in model_ids:
            auc = df_oof_metrics[
                (df_oof_metrics.model_id == model_id) & (df_oof_metrics.task == task)
            ]["auc"].values[0]
            weight_per_model_and_task.append(
                dict(
                    model_id=model_id,
                    task=task,
                    weight=np.exp(alpha * (auc - max_auc)) / sum_exp_auc,
                )
            )

    df_oof_metrics = df_oof_metrics.merge(
        pd.DataFrame(weight_per_model_and_task), on=["model_id", "task"]
    )

    print("OOF metrics per model (and model weights):")
    print(df_oof_metrics.sort_values(["task", "auc"], ascending=[True, False]))

    # generate ensemble predictions
    df_predictions = pd.concat(
        [
            predict(model_id=model_id, fold=fold, ranks=True).assign(
                model_id=model_id, fold=fold
            )
            for model_id in model_ids
            for fold in folds
        ]
    )
    df_true = df_predictions[["mol_id", "task", "y_true"]].drop_duplicates()
    df_predictions = (
        df_predictions.merge(
            df_oof_metrics[["model_id", "task", "weight"]], on=["model_id", "task"]
        )
        .groupby(["mol_id", "task"])
        .apply(lambda df: np.sum(df["y_pred"] * df["weight"]), include_groups=False)
        .to_frame(name="y_pred")
        .reset_index()
        .merge(df_true, on=["mol_id", "task"])
    )

    print("Ensemble OOF metrics:")
    print(eval(df_predictions))

    # generate submission predictions
    print("Predict test set")
    df_predictions_submission = pd.concat(
        [
            predict(
                model_id=model_id, fold=fold, subset="submission", ranks=True
            ).assign(model_id=model_id, fold=fold)
            for model_id in model_ids
            for fold in folds
        ]
    )
    df_predictions_submission = (
        df_predictions_submission.merge(
            df_oof_metrics[["model_id", "task", "weight"]], on=["model_id", "task"]
        )
        .groupby(["mol_id", "task"])
        .apply(lambda df: np.sum(df["y_pred"] * df["weight"]), include_groups=False)
        .to_frame(name="y_pred")
        .reset_index()
    )

    print("Generate submission.csv")
    df_submission = predictions_to_submission(df_predictions_submission).to_csv(
        PROJECT_ROOT / "submission.csv", index=False
    )
