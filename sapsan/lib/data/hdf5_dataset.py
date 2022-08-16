"""
HDF5 dataset classes

Usage:
    data_loader = HDF5Dataset(path="/path/to/data.h5",
                      features=['a', 'b'],
                      target=['c'],
                      checkpoints=[0.0, 0.01],
                      batch_size=BATCH_SIZE,
                      input_size=INPUT_SIZE,
                      sampler=sampler,
                      flat = False)

    x, y = data_loader.load_numpy()
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import h5py as h5
import warnings

from sapsan.core.models import Dataset, Sampling
from sapsan.utils.shapes import split_data_by_batch
from .data_functions import torch_splitter, flatten

class HDF5Dataset(Dataset):
    def __init__(self,
                 path: str,
                 input_size,
                 checkpoints: List[int] = [0],
                 features: List[str]=["None"],
                 target: List[str]=["None"],
                 batch_size: int = None,
                 batch_num: int = 1,
                 sampler: Optional[Sampling] = None,
                 time_granularity: float = 1,
                 features_label: Optional[List[str]] = ["None"],
                 target_label: Optional[List[str]] = ["None"],
                 flat: bool = False,
                 shuffle: bool = False,
                 train_fraction = None):

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
        self.batch_num = batch_num
        self.sampler = sampler
        self.input_size = input_size
        self.initial_size = input_size
        try: self.axis = len(self.input_size)
        except: self.axis = 1
        self.flat = flat
        self.shuffle = shuffle
        self.train_fraction = train_fraction

        if sampler:
            self.input_size = self.sampler.sample_dim                    
                    
        if self.batch_size!=None and self.batch_num==None:             
            self.batch_num = int(np.prod(np.array(self.input_size))/np.prod(np.array(self.batch_size)))
        if self.batch_size==None: self.batch_size = self.input_size
        if self.batch_num==None:             
            self.batch_num = len(self.checkpoints)

        #if 'target' is not a part of the file name and target_label is given = still load target
        if self.target_label!=["None"] and self.target==["None"]:
            self.target = ["None"]
                        
        self.time_granularity = time_granularity
    
    def get_parameters(self):
        parameters = {
            "data - path": self.path,
            "data - features": str(self.features)[1:-1].replace("'",""),
            "data - target": str(self.target)[1:-1].replace("'",""),
            "data - features_label": self.features_label,
            "data - target_label": self.target_label,
            "data - axis": self.axis,
            "data - shuffle": self.shuffle,
            "chkpnt - time": self.checkpoints,
            "chkpnt - initial size": self.initial_size,
            "chkpnt - sample to size": self.input_size,
            "chkpnt - time_granularity": self.time_granularity,
            "chkpnt - batch_size": self.batch_size,
            "chkpnt - batch_num" : self.batch_num
        }
        return parameters
    
    
    def load_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        #return loaded data as a numpy array only
        return self._load_data_numpy()
    
    def convert_to_torch(self, loaders: np.ndarray):
        #split into batches and convert numpy to torch dataloader
        loaders = torch_splitter(loaders, 
                                 batch_num = self.batch_num, 
                                 train_fraction = self.train_fraction,
                                 shuffle = self.shuffle)
        return loaders
    
    def load(self):
        #load numpy, split into batches, convert to torch dataloader, and return it        
        loaders = self.load_numpy()
        return self.convert_to_torch(loaders)                
    
        
    def split_batch(self, input_data):
        # columns_length ex: 12 features * 3 dim = 36  
        columns_length = input_data.shape[0]
        if self.axis == 3 or self.axis==2:
            return split_data_by_batch(input_data, self.input_size,
                                      self.batch_size, columns_length, self.axis)
    
    def _get_path(self, checkpoint, feature):
        """Return absolute path to required feature at specific checkpoint."""
        timestep = self.time_granularity * checkpoint
        relative_path = self.path.format(checkpoint=timestep, feature=feature)
        return relative_path

    
    def _get_input_data(self, checkpoint, features, labels):
        all_data = []
        # combine all features into cube with channels

        columns = max(len(features),len(labels))
        ind = np.argmax([len(features),len(labels)])
        for col in range(columns):            
            if ind==0:   features_ind = col
            elif ind==1: features_ind = 0
            
            file = h5.File(self._get_path(checkpoint, features[features_ind]), 'r')
            
            if labels==["None"]: key = list(file.keys())[-1]
            else: key = labels[col]

            print("Loading '%s' from file '%s'"%(key, self._get_path(checkpoint, features[features_ind])))
            
            data = file.get(key)

            if (self.axis==3 and len(np.shape(data))==5) or (self.axis==2 and len(np.shape(data))==4):
                warnings.warn("Warning: combining axis for %s"%key, stacklevel=2)
                data = np.reshape(data, np.append(data.shape[0]*data.shape[1],data.shape[2:]))
            
            if (self.axis==3 and len(np.shape(data))==3) or (self.axis==2 and len(np.shape(data))==2): 
                data = [data]     
            all_data.append(data)
            print('----------')
            
        # input_data shape ex: (features, 128, 128, 128) 
        input_data = np.vstack(all_data)

        # downsample if needed
        if self.sampler:
            input_data = self.sampler.sample(input_data)
            self.input_size = input_data.shape[1:]

            if np.array(self.batch_size).any != np.array(self.input_size).any:   
                warnings.warn("batch_size != sampled_size. Setting the two equal.", stacklevel=2)
                self.batch_size = self.input_size

        if self.flat: return flatten(input_data)
        elif self.batch_size == self.input_size: return np.array([input_data])
        elif len(input_data.shape)==(self.axis+2):             
            nsnaps_to_use = self._check_batch_num(input_data.shape)
            input_data = input_data[:nsnaps_to_use]
            self.batch_size = self.input_size
            return input_data
        else: 
            self._check_batch_size()
            return self.split_batch(input_data)


    def _load_data_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        print('Features: ',self.features)
        print('Fetures_label:',self.features_label)

        for i, checkpoint in enumerate(self.checkpoints):
            features_checkpoint_batch = self._get_input_data(checkpoint,
                                                             self.features,
                                                             self.features_label)
            
            if i == 0: x = features_checkpoint_batch
            else: x = np.vstack((x, features_checkpoint_batch))
                                        
            if self.target!=["None"]:
                target_checkpoint_batch = self._get_input_data(checkpoint,
                                                               self.target,
                                                               self.target_label)
                if i == 0: y = target_checkpoint_batch
                else: y = np.vstack((y,target_checkpoint_batch))                       
                
        if self.target!=["None"]: return x, y
        else: return x
    
    
    def _check_batch_size(self):
        if self.batch_size == None:
            single_batch_dim = (np.prod(self.input_size)/self.batch_num)**(1/self.axis)
            single_batch_dim = np.around(single_batch_dim, decimals=6)
            if single_batch_dim.is_integer() == False: 
                raise ValueError('Incorrect number of batches - input data cannot be evenly split')
            self.batch_size = []
            for i in range(self.axis):
                self.batch_size.append(int(single_batch_dim))    
        else: return
        
        
    def _check_batch_num(self, shape):
        nsnaps = shape[0]        
        nsnaps_to_use = nsnaps - nsnaps%self.batch_num
        
        if nsnaps_to_use != nsnaps: 
            warnings.warn("Only %d snapshots will be used, instead of %d. Adjust 'batch_num'."%(nsnaps_to_use, nsnaps), stacklevel=2)
            
        return nsnaps_to_use
            
