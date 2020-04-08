from typing import List, Tuple, Dict, Optional
import numpy as np
import h5py
from skimage.util.shape import view_as_blocks
from sklearn.model_selection import train_test_split
from torch import from_numpy
from torch.utils.data import DataLoader, TensorDataset

from sapsan.core.models import Dataset, DatasetPlugin, Sampling
from sapsan.utils.shapes import split_cube_by_grid


class FlattenFrom3dDataset(Dataset):
    _CHECKPOINT_FOLDER_NAME_PATTERN = "mhd128_t{checkpoint:.4f}/fm30/{feature}_dim128_fm30.h5"

    def __init__(self,
                 path: str,
                 features: List[str],
                 labels: List[str],
                 checkpoints: List[int],
                 sampler: Optional[Sampling] = None,
                 label_channels: Optional[List[int]] = None,
                 time_granularity: float = 2.5e-3):
        """
        @param path:
        @param features:
        @param labels:
        @param checkpoints:
        @param sampler: sampler for data
        @param label_channel: channel from labels to pick
        """
        self.path = path
        self.features = features
        self.labels = labels
        self.checkpoints = checkpoints
        self.sampler = sampler
        if sampler:
            self.checkpoint_data_size = self.sampler.sample_dim
        self.label_channels = label_channels
        self.time_granularity = time_granularity

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._load_data()

    def _get_path(self, checkpoint, feature):
        """Return absolute path to required feature at specific checkpoint."""
        timestep = self.time_granularity * checkpoint
        relative_path = self._CHECKPOINT_FOLDER_NAME_PATTERN.format(checkpoint=timestep,
                                                                    feature=feature)
        return "{0}/{1}".format(self.path, relative_path)

    def _get_checkpoint_data(self, checkpoint, columns, channels: Optional[List[int]] = None):
        all_data = []
        for col in columns:
            data = h5py.File(self._get_path(checkpoint, col), 'r')
            key = list(data.keys())[-1]
            data = data[key]
            data = np.moveaxis(data, -1, 0)
            all_data.append(data)

        checkpoint_data = np.vstack(all_data)
        # sample data if sampler is passed
        if self.sampler:
            checkpoint_data = self.sampler.sample(checkpoint_data)

        # select only required channels
        if channels:
            checkpoint_data_shape = checkpoint_data.shape
            channels_data = []
            for channel in channels:
                channels_data.append(
                    checkpoint_data[channel, :, :, :].reshape(
                        1, checkpoint_data_shape[1], checkpoint_data_shape[2], checkpoint_data_shape[3]
                    )
                )
            checkpoint_data = np.vstack(channels_data)

        columns_length = checkpoint_data.shape[0]

        return checkpoint_data.reshape(columns_length, -1)

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        x = list()
        y = list()
        for checkpoint in self.checkpoints:
            features_checkpoint_batch = self._get_checkpoint_data(checkpoint, self.features)
            labels_checkpoint_batch = self._get_checkpoint_data(checkpoint, self.labels, self.label_channels)
            x.append(features_checkpoint_batch)
            y.append(labels_checkpoint_batch)

        return np.hstack(x), np.hstack(y)
