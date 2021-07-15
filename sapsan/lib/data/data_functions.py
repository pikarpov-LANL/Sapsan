from typing import List, Tuple, Dict, Optional
import numpy as np
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from torch import from_numpy
from torch.utils.data import DataLoader, TensorDataset
from sapsan.core.models import Dataset, DatasetPlugin
     

def torch_splitter(x, y, 
                   batch_num: int,
                   train_fraction = None,
                   shuffle: bool = False):
    
    if train_fraction != None:
            x_train, x_valid, y_train, y_valid = train_test_split(x, y,
                                                                  train_size=train_fraction,
                                                                  shuffle=shuffle)
    else:
        x_train = x_valid = x
        y_train = y_valid = y

    print(np.shape(x_train), np.shape(y_train))
    dataset=TensorDataset(from_numpy(x_train).float(),
                                                    from_numpy(y_train).float())
    train_loader = DataLoader(dataset=TensorDataset(from_numpy(x_train).float(),
                                                    from_numpy(y_train).float()),
                              batch_size=batch_num,
                              shuffle=shuffle,
                              num_workers=4)

    valid_loader = DataLoader(dataset=TensorDataset(from_numpy(x_valid).float(),
                                                  from_numpy(y_valid).float()),
                            batch_size=batch_num,
                            shuffle=shuffle,
                            num_workers=4)
    print('Train data shapes: ', x_train.shape, y_train.shape)
    print('Valid data shapes: ', x_valid.shape, y_valid.shape)
    
    return OrderedDict({"train": train_loader, "valid": valid_loader})    
    
    
def flatten(data: np.ndarray):
    return data.reshape(data.shape[0], -1)

    
def get_loader_shape(loaders, name = None):
    # Get shape of the Pytorch Dataloader based on the first dataset: 'train' by default
    if name == None:
        name = next(iter(loaders))
    else: pass     
    
    x, y = iter(loaders['%s'%name]).next()
    
    return x.shape, y.shape