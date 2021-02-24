import numpy as np
from skimage.util import view_as_blocks
from logging import warning
from typing import List, Tuple, Dict, Optional

def split_cube_by_batch(data: np.ndarray,
                       size: int,
                       batch_size: int,
                       n_features: int) -> np.ndarray:
    """ --3D-- Splits big cube into smaller ones into batches.

    @param data: (channels, batch_size, batch_size, batch_size)
    @return (batch, channels, batch_size, batch_size, batch_size)
    """    
    batch = int(size ** 3 / batch_size ** 3)
    return view_as_blocks(data, block_shape=(n_features, batch_size,
                                             batch_size, batch_size)
                          ).reshape(batch, n_features,
                                    batch_size, batch_size, batch_size)

def split_square_by_batch(data: np.ndarray,
                       size: int,
                       batch_size: int,
                       n_features: int) -> np.ndarray:
    """ --2D-- Splits big square into smaller ones into batches.

    @param data: (channels, batch_size, batch_size)
    @return (batch, channels, batch_size, batch_size)
    """
    batch = int(size ** 2 / batch_size ** 2)
    return view_as_blocks(data, block_shape=(n_features, batch_size,
                                             batch_size)
                          ).reshape(batch, n_features,
                                    batch_size, batch_size)


def combine_cubes(cubes: np.ndarray,
                  checkpoint_size: int,
                  batch_size: int) -> np.ndarray:
    """ Combines batches into one big cube.

    Reverse of split_cube_by_batch function.
    @param cubes: (batch, channels, batch_size, batch_size, batch_size)
    """
    n_per_dim = int(checkpoint_size / batch_size)
    x = []
    for i in range(n_per_dim):
        y = []
        for j in range(n_per_dim):
            z = []
            for k in range(n_per_dim):
                z.append(cubes[i * n_per_dim * n_per_dim + j * n_per_dim + k])
            y.append(np.concatenate(z, axis=3))
        x.append(np.concatenate(y, axis=2))
    return np.concatenate(x, axis=1)


def slice_of_cube(data: np.ndarray,
                  feature: Optional[int] = None,
                  n_slice: Optional[int] = None):
    """ Slice of 3d cube

    @param data: numpy array
    @param feature: feature of cube to use in case of multifeature plot
    @param n_slice: slice to use
    @return: pyplot object
    """
    if len(data.shape) not in [3, 4]:
        return None

    if len(data.shape) == 4:
        if feature is None:
            warning("Feature was not provided. First one will be used")
            feature = 0
        data = data[feature, :, :, :]

    if n_slice is None:
        warning("Slice is not selected first one will be used")
        n_slice = 0

    slice_to_plot = data[n_slice]

    return slice_to_plot