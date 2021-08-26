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
from sapsan.lib.estimator.torch_backend import TorchBackend
from sapsan.lib.data import get_loader_shape

class {name_upper}Model(torch.nn.Module):
    # input channels, output channels can be the input to define the layers
    def __init__(self):
        super({name_upper}Model, self).__init__()
        
        # define your layers
        """
        self.layer_1 = torch.nn.Linear(4, 8)
        self.layer_2 = torch.nn.Linear(8, 16)
        """

    def forward(self, x): 

        # set the layer order here
        
        """
        l1 = self.layer_1(x)
        output = self.layer_2(l1)
        """

        return output
    
    
class {name_upper}Config(EstimatorConfig):
    
    # set defaults to your liking, add more parameters
    def __init__(self,
                 n_epochs: int = 1,
                 batch_dim: int = 64,
                 patience: int = 10,
                 min_delta: float = 1e-5, 
                 logdir: str = "./logs/",
                 lr: float = 1e-3,
                 min_lr = None,                 
                 *args, **kwargs):
        self.n_epochs = n_epochs
        self.batch_dim = batch_dim
        self.logdir = logdir
        self.patience = patience
        self.min_delta = min_delta
        self.lr = lr
        if min_lr==None: self.min_lr = lr*1e-2
        else: self.min_lr = min_lr
        self.kwargs = kwargs
        
        #everything in self.parameters will get recorded by MLflow
        #by default, all 'self' variables will get recorded
        self.parameters = {{f'model - {{k}}': v for k, v in self.__dict__.items() if k != 'kwargs'}}
        if bool(self.kwargs): self.parameters.update({{f'model - {{k}}': v for k, v in self.kwargs.items()}})
    
    
class {name_upper}(TorchBackend):
    # Set your optimizer, loss function, and scheduler here
    
    def __init__(self, loaders,
                       config = {name_upper}Config(), 
                       model = {name_upper}Model()):
        super().__init__(config, model)
        self.config = config
        self.loaders = loaders
        
        #uncomment if you need dataloader shapes for model input
        #x_shape, y_shape = get_shape(loaders)
        
        self.model = {name_upper}Model()
        self.optimizer = """ optimizer """
        self.loss_func = """ loss function """
        self.scheduler = """ scheduler """        
        
    def train(self):
        
        trained_model = self.torch_train(self.loaders, self.model, 
                                         self.optimizer, self.loss_func, self.scheduler, 
                                         self.config)
                
        return trained_model

'''


def get_template(name: str):
    return TEMPLATE.format(name=name.lower(),
                           name_upper=name.capitalize())
