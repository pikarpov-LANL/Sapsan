"""
HDF5 dataset classes

Usage:
    data_loader = HDF5Dataset(path="/path/to/data.h5",
                      features=['a', 'b'],
                      target=['c'],
                      checkpoints=[0.0, 0.01],
                      batch_size=BATCH_SIZE,
                      checkpoint_data_size=CHECKPOINT_DATA_SIZE,
                      sampler=sampler,
                      axis = AXIS,
                      flat = False)

    x, y = data_loader.load()
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import h5py as h5
from skimage.util.shape import view_as_blocks
from sklearn.model_selection import train_test_split
from torch import from_numpy
from torch.utils.data import DataLoader, TensorDataset

from sapsan.core.models import Dataset, DatasetPlugin, Sampling, ExperimentBackend
from sapsan.utils.shapes import split_cube_by_batch, split_square_by_batch


class HDF5DatasetPyTorchSplitterPlugin(DatasetPlugin):
    def __init__(self,
                 batch_size: int,
                 train_size: float = 0.5,
                 shuffle: bool = True):
        self.batch_size = batch_size
        self.train_size = train_size
        self.shuffle = shuffle

    def apply_on_x_y(self, x, y) -> Dict[str, DataLoader]:
        #x_train, x_test, y_train, y_test = train_test_split(x, y,
        #                                                    train_size=self.train_size,
        #                                                    shuffle=True)

        x_train = x_test = x
        y_train = y_test = y
        train_loader = DataLoader(dataset=TensorDataset(from_numpy(x_train).float(),
                                                        from_numpy(y_train).float()),
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle,
                                  num_workers=4)

        val_loader = DataLoader(dataset=TensorDataset(from_numpy(x_test).float(),
                                                      from_numpy(y_test).float()),
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                num_workers=4)

        return {"train": train_loader, "valid": val_loader}

    def apply(self, dataset: Dataset) -> Dict[str, DataLoader]:
        x, y = dataset.load()
        return self.apply_on_x_y(x, y)


class OutputFlatterDatasetPlugin(DatasetPlugin):

    @staticmethod
    def _flatten_output(output: np.ndarray):
        return output.reshape(output.shape[0], -1)

    def apply(self, dataset: Dataset):
        x, y = dataset.load()
        return x, self._flatten_output(y)

    def apply_on_x_y(self, x, y):
        return x, self._flatten_output(y)


class HDF5Dataset(Dataset):
    def __init__(self,
                 path: str,
                 features: List[str],
                 target: List[str],
                 checkpoints: List[int],
                 batch_size: int = None,
                 checkpoint_data_size: int = None,
                 sampler: Optional[Sampling] = None,
                 time_granularity: float = 1,
                 features_label: Optional[List[str]] = None,
                 target_label: Optional[List[str]] = None,
                 axis: int = 3,
                 flat: bool = False):
        """
        @param path:
        @param features:
        @param target:
        @param checkpoints:
        @param batch_size: size of cube that will be used to separate checkpoint data
        """
        self.path = path
        self.features = features
        self.target = target
        self.features_label = features_label
        self.target_label = target_label
        self.checkpoints = checkpoints
        self.batch_size = batch_size
        self.sampler = sampler
        self.checkpoint_data_size = checkpoint_data_size
        self.initial_size = checkpoint_data_size
        self.axis = axis
        self.flat = flat

        if sampler:
            self.checkpoint_data_size = self.sampler.sample_dim
        self.time_granularity = time_granularity
    
    def get_parameters(self):
        parameters = {
            "data - path": self.path,
            "data - features": str(self.features)[1:-1].replace("'",""),
            "data - target": str(self.target)[1:-1].replace("'",""),
            "data - features_label": self.features_label,
            "data - target_label": self.target_label,
            "data - axis": self.axis,
            "chkpnt - time": self.checkpoints,
            "chkpnt - initial size": self.initial_size,
            "chkpnt - sample to size": self.checkpoint_data_size,
            "chkpnt - time_granularity": self.time_granularity,
            "chkpnt - batch size": self.batch_size
        }
        return parameters
        
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._load_data()

    def _get_path(self, checkpoint, feature):
        """Return absolute path to required feature at specific checkpoint."""
        timestep = self.time_granularity * checkpoint
        relative_path = self.path.format(checkpoint=timestep, feature=feature)
        return relative_path

    def _get_checkpoint_data(self, checkpoint, columns, labels):
        all_data = []
        # combine all features into cube with channels
                
        for col in range(len(columns)):
            file = h5.File(self._get_path(checkpoint, columns[col]), 'r')
            
            if labels==None: key = list(file.keys())[-1]
            else: key = label[col]

            print("Loading '%s' from file '%s'"%(key, self._get_path(checkpoint, columns[col])))
            
            data = file.get(key)
            
            if (self.axis==3 and len(np.shape(data))==3) or (self.axis==2 and len(np.shape(data))==2): 
                data = [data]     
            all_data.append(data)
            
        # checkpoint_data shape ex: (features, 128, 128, 128)        
        checkpoint_data = np.vstack(all_data)

        # downsample if needed
        if self.sampler:
            checkpoint_data = self.sampler.sample(checkpoint_data)
            
        if self.flat: return self.flatten(checkpoint_data)
        else: return self.split_batch(checkpoint_data)


    def flatten(self, checkpoint_data):
        if self.axis == 3:
            cd_shape = checkpoint_data.shape
            return checkpoint_data.reshape(cd_shape[0],
                                           cd_shape[1]*cd_shape[2]*cd_shape[3])
        if self.axis == 2:
            cd_shape = checkpoint_data.shape
            return checkpoint_data.reshape(cd_shape[0], 
                                           cd_shape[1]*cd_shape[2])
        
    def split_batch(self, checkpoint_data):
        # columns_length ex: 12 features * 3 dim = 36  
        columns_length = checkpoint_data.shape[0]
        if self.axis == 3:
            return split_cube_by_batch(checkpoint_data, self.checkpoint_data_size,
                                      self.batch_size, columns_length)
        if self.axis == 2:
            return split_square_by_batch(checkpoint_data, self.checkpoint_data_size,
                                      self.batch_size, columns_length)

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        x = list()
        y = list()
        for checkpoint in self.checkpoints:
            features_checkpoint_batch = self._get_checkpoint_data(checkpoint, 
                                                                  self.features, self.features_label)
            target_checkpoint_batch = self._get_checkpoint_data(checkpoint, 
                                                                self.target, self.target_label)
            
            x.append(features_checkpoint_batch)
            y.append(target_checkpoint_batch)

        return np.vstack(x), np.vstack(y)
