TEMPLATE = '''
"""
Estimator Template

Please replace everything between triple quotes to create
your custom estimator.
"""
import json
import numpy as np

import torch

from sapsan.core.models import EstimatorConfig
from sapsan.lib.estimator.cnn.pytorch_estimator import TorchEstimator


class """AlgorithmNameModel"""(torch.nn.Module):
    # input channels, output channels
    def __init__(self, D_in, D_out):
        super("""AlgorithmNameModel""", self).__init__()
        
        # define your layers
        """
        self.layer_1 = torch.nn.Linear(D_in*4, D_in*8)
        self.layer_2 = torch.nn.Linear(D_in*8, D_out)
        """

    def forward(self, x): 

        # set the layer order here
        
        """
        l1 = self.layer_1(x)
        output = self.layer_2(l1)
        """

        return output
    
    
class """AlgorithmNameConfig"""(EstimatorConfig):
    
    # set defaults per your liking, add more parameters
    def __init__(self,
                 n_epochs: int = 1,
                 batch_dim: int = 64,
                 patience: int = 10,
                 min_delta: float = 1e-5, 
                 logdir: str = "./logs/",
                 *args, **kwargs):
        self.n_epochs = n_epochs
        self.batch_dim = batch_dim
        self.logdir = logdir
        self.patience = patience
        self.min_delta = min_delta
        
        #a few custom names for parameters to record in mlflow
        self.parameters = {{
                        "model - n_epochs": self.n_epochs,
                        "model - min_delta": self.min_delta,
                        "model - patience": self.patience,
                    }}

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            cfg = json.load(f)
            return cls(**cfg)

    def to_dict(self):
        return self.parameters    
    
    
class """AlgorithmName"""(TorchEstimator):
    def __init__(self, config: """AlgorithmNameConfig""", model=None):
        super().__init__(config, model)
        self.config = config
        self.model = """AlgorithmNameModel"""(1, 1)
        
    def setup_model(self, n_input_channels, n_output_channels):
        return """AlgorithmNameModel"""(n_input_channels, self.config.batch_dim ** 3 * n_output_channels)

    def train(self, inputs, targets=None):

        self.model = self.setup_model(inputs.shape[1], targets.shape[1])
        optimizer = """ optimizer """
        loss_func = """ loss finctions """
        scheduler = """ scheduler """
        
        model = self.torch_train(inputs, targets, 
                                 self.model, optimizer, loss_func, scheduler, self.config)
                
        return model

'''


def get_template(name: str):
    return TEMPLATE.format(name=name.lower(),
                           name_upper=name.capitalize())
