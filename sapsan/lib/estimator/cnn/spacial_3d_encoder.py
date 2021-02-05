"""
Spacial 3d encoder estimator
Example:
    estimator = CNN3d(
        config=CNN3dConfig(n_epochs=1)
    )

    ds = JHTDBDataset(path="/Users/icekhan/Documents/development/myprojects/sapsan/repo/Sapsan/dataset",
                      features=['u', 'b', 'a',
                                'du0', 'du1', 'du2',
                                'db0', 'db1', 'db2',
                                'da0', 'da1', 'da2'],
                      labels=['tn'],
                      checkpoints=[0.0, 0.01, 0.025, 0.25])

    x, y = ds.load()

    model = estimator.train(x, y)
"""
import json
from typing import Dict
import numpy as np

import torch
from catalyst.dl import SupervisedRunner, EarlyStoppingCallback, CheckpointCallback, IterationCheckpointCallback

from sapsan.lib.data.hdf5_dataset import HDF5DatasetPyTorchSplitterPlugin, OutputFlatterDatasetPlugin
from sapsan.core.models import Estimator, EstimatorConfig


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

        torch.cuda.empty_cache()
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

class SkipCheckpointCallback(CheckpointCallback):
    def on_epoch_end(self, state):
        pass
    
class CNN3d(Estimator):

    def __init__(self, config: CNN3dConfig):
        super().__init__(config)

        self.config = config
        self.model = CNN3dModel(1, 1) #perhaps we want to re-think our save-load test; no need to load the model here
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.runner = SupervisedRunner()
        self.model_metrics = dict()

    def setup_model(self, n_input_channels, n_output_channels):
        self.model = CNN3dModel(n_input_channels, self.config.grid_dim ** 3 * n_output_channels)
        
    def predict(self, inputs):
        if str(self.device) == 'cpu': data = torch.as_tensor(inputs)
        else: data = torch.as_tensor(inputs).cuda()
        
        return self.model(data).cpu().data.numpy()

    def metrics(self) -> Dict[str, float]:
        return self.model_metrics
        
    def train(self, inputs, targets=None):
        
        print('Device used:', self.device)
        
        self.setup_model(inputs.shape[1], targets.shape[1])
        
        output_flatter = OutputFlatterDatasetPlugin()
        splitter_pytorch = HDF5DatasetPyTorchSplitterPlugin(4)
        _, flatten_targets = output_flatter.apply_on_x_y(inputs, targets)
        loaders = splitter_pytorch.apply_on_x_y(inputs, flatten_targets)

        model = self.model
        if torch.cuda.device_count() > 1:
            print("GPUs available: ", torch.cuda.device_count())
            print("Note: if batch_size == 1, then only 1 GPU will be used")
            model = torch.nn.DataParallel(model)
        
        model.to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_func = torch.nn.MSELoss()  # torch.nn.SmoothL1Loss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=3,
            min_lr=1e-5) 

        torch.cuda.empty_cache()
        self.runner.train(model=model,
                          criterion=loss_func,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          loaders=loaders,
                          logdir=self.config.logdir,
                          num_epochs=self.config.n_epochs,
                          callbacks=[EarlyStoppingCallback(patience=self.config.patience,
                                                           min_delta=self.config.min_delta),
                                    SkipCheckpointCallback()
                                    ],
                          verbose=False,
                          check=False)
        
        self.config.parameters['model - device'] = self.runner.device         
        self.model_metrics['final epoch'] = self.runner.epoch-1
        for key,value in self.runner.epoch_metrics.items():
            self.model_metrics[key] = value

        with open('model_details.txt', 'w') as file:
            file.write('%s\n\n%s\n\n%s'%(str(self.runner.model),
                                   str(self.runner.optimizer),
                                   str(self.runner.scheduler)))
        
        return model

    def save(self, path):
        model_save_path = "{path}/model".format(path=path)
        params_save_path = "{path}/params.json".format(path=path)

        torch.save(self.model.state_dict(), model_save_path)
        self.config.save(params_save_path)

    @classmethod
    def load(cls, path: str):
        model_save_path = "{path}/model".format(path=path)
        params_save_path = "{path}/params.json".format(path=path)

        config = CNN3dConfig.load(params_save_path)

        estimator = CNN3d(config)
        model = estimator.model.load_state_dict(torch.load(model_save_path))
        # model.eval()
        estimator.model = model
        return estimator