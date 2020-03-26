import numpy as np
from sapsan.core.models import Sampling


class Equidistance3dSampling(Sampling):
    def __init__(self, original_dim: int, target_dim: int):
        self.original_dim = original_dim
        self.target_dim = target_dim

    @property
    def scale(self):
        return int(self.original_dim / self.target_dim)

    @property
    def sample_dim(self):
        return self.target_dim

    def sample(self, data: np.ndarray):
        return data[:, ::self.scale, ::self.scale, ::self.scale]