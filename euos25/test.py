import torch

from .data import Tox25Dataset

#!ignore
subset = "train"
mode = "continuous"

train_set = "full_train_fold_4"
val_set = "full_test_fold_4"

dataset_train = Tox25Dataset(
    label=None,
    subset=train_set,
    mode=mode,
    use_plate_indices=True,
    normalize_by_plate=False,
    normalize_labels=True,
)
dataset_val = Tox25Dataset(
    label=None,
    subset=val_set,
    mode="binary",
)
print(len(dataset_train), len(dataset_val))

# check for nan in dataset_train
for i in dataset_train:
    if torch.isnan(i.y).any():
        print("Found NaN in y:", i, i.y)
        break

for i in range(3):
    print(
        dataset_train[i],
        dataset_train[i].y,
        dataset_train[i].plate_id,
        dataset_train[i].label_type,
        dataset_train[i].task_id,
    )
