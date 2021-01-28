import numpy as np
from skimage.util import view_as_blocks


def split_cube_by_grid(data: np.ndarray,
                       size: int,
                       grid_size: int,
                       n_features: int) -> np.ndarray:
    """ --3D-- Splits big cube into smaller ones into grid.

    @param data: (channels, grid_size, grid_size, grid_size)
    @return (batch, channels, grid_size, grid_size, grid_size)
    """    
    batch = int(size ** 3 / grid_size ** 3)
    return view_as_blocks(data, block_shape=(n_features, grid_size,
                                             grid_size, grid_size)
                          ).reshape(batch, n_features,
                                    grid_size, grid_size, grid_size)

def split_square_by_grid(data: np.ndarray,
                       size: int,
                       grid_size: int,
                       n_features: int) -> np.ndarray:
    """ --2D-- Splits big square into smaller ones into grid.

    @param data: (channels, grid_size, grid_size)
    @return (batch, channels, grid_size, grid_size)
    """
    batch = int(size ** 2 / grid_size ** 2)
    return view_as_blocks(data, block_shape=(n_features, grid_size,
                                             grid_size)
                          ).reshape(batch, n_features,
                                    grid_size, grid_size)


def combine_cubes(cubes: np.ndarray,
                  checkpoint_size: int,
                  grid_size: int) -> np.ndarray:
    """ Combines cubes in a grid into one big cube.

    Reverse of split_cube_by_grid function.
    @param cubes: (batch, channels, grid_size, grid_size, grid_size)
    """
    n_per_dim = int(checkpoint_size / grid_size)
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
