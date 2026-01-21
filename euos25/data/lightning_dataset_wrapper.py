import copy

from torch.utils.data import DataLoader
from torch_geometric.data.lightning import LightningDataset


class LightningDatasetWrapper(LightningDataset):
    def __init__(self, *args, **kwargs):
        if "batch_sampler_generator" in kwargs:
            self.batch_sampler_generator = kwargs["batch_sampler_generator"]
            kwargs.pop("batch_sampler_generator")
        else:
            self.batch_sampler_generator = None
        super().__init__(
            *args,
            **kwargs,
        )

    # we are patching train_dataloader here to add a dynamic batch_sampler
    def train_dataloader(self) -> DataLoader:
        from torch.utils.data import IterableDataset

        kwargs = copy.copy(self.kwargs)

        batch_sampler = kwargs.get("batch_sampler", None)
        if batch_sampler is None and self.batch_sampler_generator is not None:
            batch_sampler = self.batch_sampler_generator(self.batch_size)

        shuffle = not isinstance(self.train_dataset, IterableDataset)
        shuffle &= kwargs.get("sampler", None) is None
        shuffle &= batch_sampler is None

        return self.dataloader(
            self.train_dataset,
            shuffle=shuffle,
            **kwargs,
        )

    @property
    def batch_size(self):
        return self.kwargs.get("batch_size", 32)

    @batch_size.setter
    def batch_size(self, value):
        self.kwargs["batch_size"] = value
