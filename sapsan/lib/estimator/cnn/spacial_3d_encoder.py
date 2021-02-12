"""
Spacial 3d encoder estimator

    - uses pytorch_estimator for backend

For example see sapsan/examples/cnn_example.ipynb
"""
import json
import numpy as np

import torch

from sapsan.core.models import EstimatorConfig
from sapsan.lib.estimator.cnn.pytorch_estimator import TorchEstimator


class CNN3dModel(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(CNN3dModel, self).__init__()
        
        self.conv3d = torch.nn.Conv3d(D_in, D_in*2, kernel_size=2, stride=2, padding=1)
        self.pool = torch.nn.MaxPool3d(kernel_size=2, padding=1)
        self.conv3d2 = torch.nn.Conv3d(D_in*2, D_in*2, kernel_size=2, stride=2, padding=1)
        self.pool2 = torch.nn.MaxPool3d(kernel_size=2, padding=1)
        self.conv3d3 = torch.nn.Conv3d(D_in*2, D_in*4, kernel_size=2, stride=2, padding=1)
        self.pool3 = torch.nn.MaxPool3d(kernel_size=2)

        self.relu = torch.nn.ReLU()

        self.linear = torch.nn.Linear(D_in*4, D_in*8)
        self.linear2 = torch.nn.Linear(D_in*8, D_out)

    def forward(self, x): 

        #torch.cuda.empty_cache()
        x = x.float()
        c1 = self.conv3d(x)
        p1 = self.pool(c1)
        c2 = self.conv3d2(self.relu(p1))
        p2 = self.pool2(c2)
        c3 = self.conv3d3(self.relu(p2))
        p3 = self.pool3(c3)

        v1 = p3.view(p3.size(0), -1)

        l1 = self.relu(self.linear(v1))
        l2 = self.linear2(l1)   

        return l2
    
    
class CNN3dConfig(EstimatorConfig):
    def __init__(self,
                 n_epochs: int = 1,
                 grid_dim: int = 64,
                 patience: int = 10,
                 min_delta: float = 1e-5, 
                 logdir: str = "./logs/",
                 *args, **kwargs):
        self.n_epochs = n_epochs
        self.grid_dim = grid_dim
        self.logdir = logdir
        self.patience = patience
        self.min_delta = min_delta
        self.parameters = {
                        "model - n_epochs": self.n_epochs,
                        "model - min_delta": self.min_delta,
                        "model - patience": self.patience,
                    }

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            cfg = json.load(f)
            return cls(**cfg)

    def to_dict(self):
        return self.parameters    
    
    
class CNN3d(TorchEstimator):
    def __init__(self, config: CNN3dConfig, model=None):
        super().__init__(config, model)
        self.config = config
        self.model = CNN3dModel(1, 1) #perhaps we want to re-think our save-load test; no need to load the model here
        
    def setup_model(self, n_input_channels, n_output_channels):
        return CNN3dModel(n_input_channels, self.config.grid_dim ** 3 * n_output_channels)

    def train(self, inputs, targets=None):

        self.model = self.setup_model(inputs.shape[1], targets.shape[1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_func = torch.nn.MSELoss()  # torch.nn.SmoothL1Loss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               patience=3,
                                                               min_lr=1e-5) 
        
        model = self.torch_train(inputs, targets, 
                                 self.model, optimizer, loss_func, scheduler, self.config)
                
        return model