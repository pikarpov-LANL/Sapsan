"""
JHTDB dataset classes

Usage:
    ds = JHTDBDataset(path="/Users/icekhan/Documents/development/myprojects/sapsan/repo/Sapsan/dataset",
                      features=['u', 'b', 'a',
                                'du0', 'du1', 'du2',
                                'db0', 'db1', 'db2',
                                'da0', 'da1', 'da2'],
                      labels=['tn'],
                      checkpoints=[0])

    plugin = JHTDBDatasetPyTorchSplitterPlugin(4)
    loaders = plugin.apply(ds)
"""

from typing import List, Tuple, Dict
import numpy as np
import h5py
from skimage.util.shape import view_as_blocks
from sklearn.model_selection import train_test_split
from torch import from_numpy
from torch.utils.data import DataLoader, TensorDataset

from sapsan.general.models import Dataset, DatasetPlugin


class JHTDBDatasetPyTorchSplitterPlugin(DatasetPlugin):
    def __init__(self,
                 batch_size: int,
                 train_size: float = 0.5,
                 shuffle: bool = True):
        self.batch_size = batch_size
        self.train_size = train_size
        self.shuffle = shuffle

    def apply(self, dataset: Dataset) -> Dict[str, DataLoader]:
        x, y = dataset.load()
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=self.train_size)

        train_loader = DataLoader(dataset=TensorDataset(from_numpy(x_train),
                                                        from_numpy(y_train)),
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle,
                                  num_workers=4)

        val_loader = DataLoader(dataset=TensorDataset(from_numpy(x_test),
                                                      from_numpy(y_test)),
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                num_workers=4)

        return {"train": train_loader, "valid": val_loader}


class JHTDBDataset(Dataset):
    _CHECKPOINT_FOLDER_NAME_PATTERN = "mhd128_t{checkpoint}.0000/fm30/{feature}_dim128_fm30.h5"
    _CHECKPOINT_DATA_SIZE = 128
    _GRID_SIZE = 64

    def __init__(self,
                 path: str,
                 features: List[str],
                 labels: List[str],
                 checkpoints: List[int]):
        """
        @param path:
        @param features:
        @param labels:
        @param checkpoints:
        """
        self.path = path
        self.features = features
        self.labels = labels
        self.checkpoints = checkpoints

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._load_data()

    def _get_path(self, checkpoint, feature):
        """Return absolute path to required feature at specific checkpoint."""
        relative_path = self._CHECKPOINT_FOLDER_NAME_PATTERN.format(checkpoint=checkpoint,
                                                                    feature=feature)
        return "{0}/{1}".format(self.path, relative_path)

    def _get_checkpoint_data(self, checkpoint, columns):
        all_data = []
        # combine all features into cube with channels
        for col in columns:
            data = h5py.File(self._get_path(checkpoint, col), 'r')
            key = list(data.keys())[-1]
            data = data[key]
            data = np.moveaxis(data, -1, 0)
            all_data.append(data)
        # checkpoint_data shape: (features, 128, 128, 128)
        checkpoint_data = np.vstack(all_data)
        # columns_length: 12 features (or 1 label) * 3 dim = 36
        columns_length = checkpoint_data.shape[0]

        checkpoint_batch_size = int(self._CHECKPOINT_DATA_SIZE ** 3 / self._GRID_SIZE ** 3)
        # checkpoint_batch shape: (batch_size, channels, 128, 128, 128)
        checkpoint_batch = view_as_blocks(checkpoint_data,
                                          block_shape=(columns_length, self._GRID_SIZE,
                                                       self._GRID_SIZE, self._GRID_SIZE)
                                          ).reshape(checkpoint_batch_size, columns_length,
                                                    self._GRID_SIZE, self._GRID_SIZE, self._GRID_SIZE)

        return checkpoint_batch

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        x = list()
        y = list()
        for checkpoint in self.checkpoints:
            features_checkpoint_batch = self._get_checkpoint_data(checkpoint, self.features)
            labels_checkpoint_batch = self._get_checkpoint_data(checkpoint, self.labels)
            x.append(features_checkpoint_batch)
            y.append(labels_checkpoint_batch)

        return np.vstack(x), np.vstack(y)
