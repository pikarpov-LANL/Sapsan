import numpy as np
from sapsan.core.models import Sampling


class EquidistantSampling(Sampling):
    def __init__(self, target_dim):
        self.target_dim = target_dim

    @property
    def scale(self):
        return int(self.original_dim[0] / self.target_dim[0])

    @property
    def sample_dim(self):
        return self.target_dim

    def dim_warning(self, new_dim):
        if list(self.target_dim) not in [new_dim, new_dim[1:]]: 
            print("Warning: couldn't cover the whole domain and sample to ", self.target_dim,
                  ", new sampled shape is ", new_dim)
    
    def sample(self, data: np.ndarray):
        self.original_dim = data.shape[-len(self.target_dim):]

        print("Sampling the input data of size", self.original_dim, "into size", self.target_dim)
        
        one_dim = self.original_dim[0]
        for i in self.original_dim:
            if i==one_dim: pass
            else: 
                raise ValueError('Error: Equidistant sampling can only be applied to axi of equal dimensions, but recieved', self.original_dim)
                return data
            one_dim = i
            
        
        if len(self.original_dim) == 3:
            data = data[..., ::self.scale, ::self.scale, ::self.scale]
        if len(self.original_dim) == 2:
            data = data[..., ::self.scale, ::self.scale]
        
        self.dim_warning(list(data.shape))
        
        return data