"""
Spacial 3d encoder estimator
Example:
    estimator = Spacial3dEncoderNetworkEstimator(
        config=Spacial3dEncoderNetworkEstimatorConfiguration(n_epochs=1)
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

import torch
from catalyst.dl import SupervisedRunner, EarlyStoppingCallback

from sapsan.lib.data.jhtdb_dataset import JHTDBDatasetPyTorchSplitterPlugin, OutputFlatterDatasetPlugin
from sapsan.core.models import Estimator, EstimatorConfiguration


class Spacial3dEncoderNetworkModel(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(Spacial3dEncoderNetworkModel, self).__init__()

        self.conv3d = torch.nn.Conv3d(D_in, 72, 2, stride=2, padding=1)
        self.pooling = torch.nn.MaxPool3d(kernel_size=2, padding=1)
        self.conv3d2 = torch.nn.Conv3d(72, 72, 2, stride=2, padding=1)
        self.pooling2 = torch.nn.MaxPool3d(kernel_size=2, padding=1)
        self.conv3d3 = torch.nn.Conv3d(72, 144, 2, stride=2, padding=1)
        self.pooling3 = torch.nn.MaxPool3d(kernel_size=2)

        self.relu = torch.nn.ReLU()

        self.linear = torch.nn.Linear(144, 288)
        self.linear2 = torch.nn.Linear(288, D_out)

    def forward(self, x):
        c1 = self.conv3d(x)
        p1 = self.pooling(c1)
        c2 = self.conv3d2(self.relu(p1))
        p2 = self.pooling2(c2)

        c3 = self.conv3d3(self.relu(p2))
        p3 = self.pooling3(c3)
        v1 = p3.view(p3.size(0), -1)

        l1 = self.relu(self.linear(v1))
        l2 = self.linear2(l1)

        return l2


class Spacial3dEncoderNetworkEstimatorConfiguration(EstimatorConfiguration):
    def __init__(self,
                 n_epochs: int,
                 n_input_channels: int = 36,
                 n_output_channels: int = 3,
                 grid_dim: int = 64,
                 logdir: str = "./logs/"):
        self.n_epochs = n_epochs
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.grid_dim = grid_dim
        self.logdir = logdir

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            cfg = json.load(f)
            return cls(**cfg)

    def to_dict(self):
        return {
            "n_epochs": self.n_epochs,
            "grid_dim": self.grid_dim,
            "n_input_channels": self.n_input_channels,
            "n_output_channels": self.n_output_channels
        }


class Spacial3dEncoderNetworkEstimator(Estimator):

    def __init__(self, config: Spacial3dEncoderNetworkEstimatorConfiguration):
        super().__init__(config)

        self.config = config

        self.model = Spacial3dEncoderNetworkModel(self.config.n_input_channels,
                                                  self.config.grid_dim ** 3 * self.config.n_output_channels)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.runner = SupervisedRunner()
        self.model_metrics = dict()

    def predict(self, inputs):
        return self.model(torch.as_tensor(inputs)).cpu().data.numpy()

    def metrics(self) -> Dict[str, float]:
        return self.model_metrics

    def train(self, inputs, targets=None):
        output_flatter = OutputFlatterDatasetPlugin()
        splitter_pytorch = JHTDBDatasetPyTorchSplitterPlugin(4)
        _, flatten_targets = output_flatter.apply_on_x_y(inputs, targets)
        loaders = splitter_pytorch.apply_on_x_y(inputs, flatten_targets)

        model = self.model
        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_func = torch.nn.MSELoss()  # torch.nn.SmoothL1Loss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=3,
            min_lr=1e-5)

        self.runner.train(model=model,
                          criterion=loss_func,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          loaders=loaders,
                          logdir=self.config.logdir,
                          num_epochs=self.config.n_epochs,
                          callbacks=[EarlyStoppingCallback(patience=10, min_delta=1e-5)],
                          verbose=False,
                          check=False)

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

        config = Spacial3dEncoderNetworkEstimatorConfiguration.load(params_save_path)

        estimator = Spacial3dEncoderNetworkEstimator(config)
        model = estimator.model.load_state_dict(torch.load(model_save_path))
        # model.eval()
        estimator.model = model
        return estimator
