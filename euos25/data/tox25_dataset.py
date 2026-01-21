import os

import numpy as np
import pandas as pd
import torch
from ogb.utils import smiles2graph
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.data import Data, InMemoryDataset

from ..paths import DATA_DIR

label_columns = ["t340", "t450", "f340", "f480"]
continous_label_columns = ["t340_quant", "t450_quant"]
subsets_folds_train = [f"train_fold_{i}" for i in range(5)]
subsets_folds_test = [f"test_fold_{i}" for i in range(5)]
subsets_folds_full_train_leaderboard = [f"full_train_fold_{i}" for i in range(5)]
subsets_folds_full_test_leaderboard = [f"full_test_fold_{i}" for i in range(5)]
subsets = (
    ["train", "test", "full_train", "leaderboard", "submission"]
    + subsets_folds_train
    + subsets_folds_test
    + subsets_folds_full_train_leaderboard
    + subsets_folds_full_test_leaderboard
)


class Tox25Dataset(InMemoryDataset):
    def __init__(
        self,
        root=DATA_DIR,
        subset="train",
        mode="binary",
        label=None,
        use_absorbance=False,
        use_plate_indices=False,
        use_plate_iterations=False,
        normalize_by_plate=False,
        normalize_labels=False,
        transform=None,
    ):
        assert subset in subsets
        assert mode in ["binary", "continuous"]
        assert label is None or label in label_columns
        assert not (use_plate_indices and use_plate_iterations)
        self.subset = subset
        self.mode = mode
        self.label = label
        self.use_absorbance = use_absorbance
        self.use_plate_indices = use_plate_indices
        self.use_plate_iterations = use_plate_iterations
        self.normalize_by_plate = normalize_by_plate
        self.normalize_labels = normalize_labels
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed")

    @property
    def raw_file_names(self):
        return [
            "derived/df_train.csv",
            "derived/df_leaderboard.csv",
            "derived/df_test.csv",
        ]

    @property
    def processed_file_names(self):
        suffix = ""

        if self.label is not None:
            suffix = suffix + f"_{self.label}"

        suffix += f"_{self.subset}_{self.mode}"

        if self.use_absorbance:
            suffix = suffix + "_with_absorbance"

        if self.use_plate_indices:
            suffix = suffix + "_with_plate_indices"
        elif self.use_plate_iterations:
            suffix = suffix + "_with_plate_iterations"

        if self.normalize_by_plate:
            suffix = suffix + "_normalized_by_plate"

        if self.normalize_labels:
            suffix = suffix + "_normalized_labels"

        return [f"data_tox25{suffix}.pt"]

    def download(self):
        pass

    def process(self):
        if self.subset == "train" or self.subset == "test":
            df = pd.read_csv(self.raw_paths[0])

            train_idx, val_idx = train_test_split(
                np.arange(len(df)),
                test_size=0.175,
                random_state=42,
            )

            if self.subset == "train":
                df = df.iloc[train_idx].reset_index(drop=True)
            else:
                df = df.iloc[val_idx].reset_index(drop=True)
        elif self.subset == "full_train":
            df = pd.read_csv(self.raw_paths[0])
        elif self.subset == "leaderboard":
            df = pd.read_csv(self.raw_paths[1])
        elif self.subset == "submission":
            df = pd.read_csv(self.raw_paths[2])
        elif self.subset in subsets_folds_train + subsets_folds_test:
            df_full_train = pd.read_csv(self.raw_paths[0])

            is_train = self.subset in subsets_folds_train

            kf = KFold(n_splits=len(subsets_folds_train), shuffle=True, random_state=42)
            folds = list(kf.split(df_full_train))

            if is_train:
                fold_idx = subsets_folds_train.index(self.subset)
            else:
                fold_idx = subsets_folds_test.index(self.subset)

            train_indices, test_indices = folds[fold_idx]

            if is_train:
                df = df_full_train.iloc[train_indices].reset_index(drop=True)
            else:
                df = df_full_train.iloc[test_indices].reset_index(drop=True)
        elif (
            self.subset
            in subsets_folds_full_train_leaderboard
            + subsets_folds_full_test_leaderboard
        ):
            df_full_train = pd.read_csv(self.raw_paths[0])
            df_leaderboard = pd.read_csv(self.raw_paths[1])

            is_train = self.subset in subsets_folds_full_train_leaderboard

            kf = KFold(
                n_splits=len(subsets_folds_full_train_leaderboard),
                shuffle=True,
                random_state=42,
            )
            folds_train = list(kf.split(df_full_train))
            folds_leaderboard = list(kf.split(df_leaderboard))

            if is_train:
                fold_idx = subsets_folds_full_train_leaderboard.index(self.subset)
            else:
                fold_idx = subsets_folds_full_test_leaderboard.index(self.subset)

            train_indices_full_train, test_indices_full_train = folds_train[fold_idx]
            train_indices_leaderboard, test_indices_leaderboard = folds_leaderboard[
                fold_idx
            ]

            if is_train:
                indices_full_train = train_indices_full_train
                indices_leaderboard = train_indices_leaderboard
            else:
                indices_full_train = test_indices_full_train
                indices_leaderboard = test_indices_leaderboard

            df_full_train_fold = df_full_train.iloc[indices_full_train].reset_index(
                drop=True
            )
            df_leaderboard_fold = df_leaderboard.iloc[indices_leaderboard].reset_index(
                drop=True
            )

            if self.mode == "binary":
                common_columns = df_full_train_fold.columns.intersection(
                    df_leaderboard_fold.columns
                )
                df = pd.concat(
                    [
                        df_full_train_fold[common_columns],
                        df_leaderboard_fold[common_columns],
                    ],
                    ignore_index=True,
                )

                # shuffle the combined dataframe
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            elif self.mode == "continuous":
                # leaderboard doesn't have continuous labels
                df = df_full_train_fold
        else:
            raise ValueError(f"Unknown subset: {self.subset}")

        data_list = []

        if self.label is not None:
            if self.label in label_columns:
                relevant_label_columns = [self.label]
            elif self.label in ["t340", "t450"]:
                relevant_label_columns = [self.label + "_quant"]
        else:
            if self.mode == "binary":
                relevant_label_columns = label_columns
            elif self.mode == "continuous":
                relevant_label_columns = continous_label_columns
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        if (
            self.use_absorbance
            and self.subset
            in ["train", "full_train"]
            + subsets_folds_train
            + subsets_folds_full_train_leaderboard
        ):
            for label in relevant_label_columns:
                if label in ["t340_quant", "t450_quant"]:
                    df[label] = df[label].apply(
                        lambda x: -np.log10(x) if x > 0 else 0.0
                    )

        if (
            self.normalize_by_plate
            and self.subset
            in ["train", "full_train"]
            + subsets_folds_train
            + subsets_folds_full_train_leaderboard
        ):
            if self.use_plate_indices:
                for label in relevant_label_columns:
                    df[label] = df.groupby("plate_index")[label].transform(
                        lambda x: (x - x.mean())
                    )
            elif self.use_plate_iterations:
                for label in relevant_label_columns:
                    df[label] = df.groupby("plate_iteration")[label].transform(
                        lambda x: (x - x.mean())
                    )

        if self.normalize_labels:
            for label in relevant_label_columns:
                df[label] = (df[label] - df[label].mean()) / df[label].std()

        task_id_mapping = dict(
            [
                (label, i)
                for i, label in enumerate(label_columns + continous_label_columns)
            ]
        )

        # iterate through rows of df
        for row in df.itertuples():
            for label in relevant_label_columns:
                task_id = task_id_mapping[label]

                mol_id = row.id
                graph = smiles2graph(row.smiles)
                if self.subset == "submission":
                    label_value = -1
                else:
                    label_value = row._asdict()[label]

                info = dict(
                    mol_id=torch.tensor([mol_id], dtype=torch.long),
                    x=torch.tensor(graph["node_feat"], dtype=torch.long),
                    edge_index=torch.tensor(
                        graph["edge_index"], dtype=torch.long
                    ).contiguous(),
                    edge_attr=torch.tensor(graph["edge_feat"], dtype=torch.float),
                    y=torch.tensor([label_value], dtype=torch.float),
                    label_type=torch.tensor(
                        [1 if label in continous_label_columns else 0],
                        dtype=torch.long,
                    ),
                )

                if len(relevant_label_columns) > 1:
                    info["task_id"] = torch.tensor([task_id], dtype=torch.long)

                if self.use_plate_indices:
                    # note: test is the validation set, so it is fine
                    if (
                        self.subset
                        in ["train", "full_train"]
                        + subsets_folds_train
                        + subsets_folds_full_train_leaderboard
                    ):
                        info["plate_id"] = torch.tensor(
                            [row.plate_index], dtype=torch.long
                        )
                    else:
                        # for test and leaderboard sets, plate indices are unknown
                        info["plate_id"] = torch.tensor([-1], dtype=torch.long)
                elif self.use_plate_iterations:
                    if (
                        self.subset
                        in ["train", "full_train"]
                        + subsets_folds_train
                        + subsets_folds_full_train_leaderboard
                    ):
                        info["plate_id"] = torch.tensor(
                            [row.plate_iteration], dtype=torch.long
                        )
                    else:
                        # for test and leaderboard sets, plate iterations are unknown
                        info["plate_id"] = torch.tensor([-1], dtype=torch.long)

                data = Data(**info)
                data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
