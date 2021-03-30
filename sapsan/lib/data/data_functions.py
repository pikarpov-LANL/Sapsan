from typing import List, Tuple, Dict, Optional
import numpy as np
from sklearn.model_selection import train_test_split
from torch import from_numpy
from torch.utils.data import DataLoader, TensorDataset
from sapsan.core.models import Dataset, DatasetPlugin

class DatasetPytorchSplitterPlugin(DatasetPlugin):
    def __init__(self,
                 num_batches: int,
                 train_fraction = None,
                 shuffle: bool = False):
        self.num_batches = num_batches
        self.train_fraction = train_fraction
        self.shuffle = shuffle

    def apply_on_x_y(self, x, y) -> Dict[str, DataLoader]:
        if self.train_fraction != None:
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                train_fraction=self.train_fraction,
                                                                shuffle=self.shuffle)
        else:
            x_train = x_test = x
            y_train = y_test = y
        train_loader = DataLoader(dataset=TensorDataset(from_numpy(x_train).float(),
                                                        from_numpy(y_train).float()),
                                  batch_size=self.num_batches,
                                  shuffle=self.shuffle,
                                  num_workers=4)

        val_loader = DataLoader(dataset=TensorDataset(from_numpy(x_test).float(),
                                                      from_numpy(y_test).float()),
                                batch_size=self.num_batches,
                                shuffle=self.shuffle,
                                num_workers=4)

        return {"train": train_loader, "valid": val_loader}

    def apply(self, dataset: Dataset) -> Dict[str, DataLoader]:
        x, y = dataset.load()
        return self.apply_on_x_y(x, y)


class FlatterDatasetPlugin(DatasetPlugin):

    @staticmethod
    def _flatten_output(output: np.ndarray):
        return output.reshape(output.shape[0], -1)

    def apply(self, dataset: Dataset):
        x, y = dataset.load()
        return x, self._flatten_output(y)

    def apply_on_x_y(self, x, y):
        return x, self._flatten_output(y)