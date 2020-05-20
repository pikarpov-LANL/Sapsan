import numpy as np
from sapsan.core.models import Sampling


class EquidistanceSampling(Sampling):
    def __init__(self, original_dim: int, target_dim: int, axis: int):
        self.original_dim = original_dim
        self.target_dim = target_dim
        self.axis = axis

    @property
    def scale(self):
        return int(self.original_dim / self.target_dim)

    @property
    def sample_dim(self):
        return self.target_dim

    def sample(self, data: np.ndarray):
        if self.axis == 3:
            return data[:, ::self.scale, ::self.scale, ::self.scale]
        if self.axis == 2:
            return data[:, ::self.scale, ::self.scale]