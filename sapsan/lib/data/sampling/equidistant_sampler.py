import numpy as np
from sapsan.core.models import Sampling


class EquidistantSampling(Sampling):
    def __init__(self, original_dim, target_dim):
        self.original_dim = original_dim
        self.target_dim = target_dim

    @property
    def scale(self):
        return int(self.original_dim[0] / self.target_dim[0])

    @property
    def sample_dim(self):
        return self.target_dim

    def sample(self, data: np.ndarray):
        
        one_dim = self.original_dim[0]
        for i in self.original_dim:
            if i==one_dim: pass
            else: 
                print('Warning: Equidistant sampling can only be applied to axi of equal dimensions. Returning original dataset.')
                return data
            one_dim = i
        
        if len(self.original_dim) == 3:
            return data[..., ::self.scale, ::self.scale, ::self.scale]
        if len(self.original_dim) == 2:
            return data[..., ::self.scale, ::self.scale]