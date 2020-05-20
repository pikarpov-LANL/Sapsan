import json
from typing import Dict

import torch
from catalyst.dl import SupervisedRunner, EarlyStoppingCallback

from sapsan.lib.data.hdf5_dataset import HDF5DatasetPyTorchSplitterPlugin
from sapsan.core.models import Estimator, EstimatorConfig


class CAEModel(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CAEModel, self).__init__()

        self.conv1 = torch.nn.Conv3d(input_channels, 72, 2, stride=2, padding=1)
        self.pool1 = torch.nn.MaxPool3d(kernel_size=2, padding=1, return_indices=True)
        self.conv2 = torch.nn.Conv3d(72, 72, 2, stride=2, padding=1)
        self.pool2 = torch.nn.MaxPool3d(kernel_size=2, padding=1, return_indices=True)

        self.unpool2 = torch.nn.MaxUnpool3d(kernel_size=2, padding=1)
        self.deconv2 = torch.nn.ConvTranspose3d(72, 72, 2, stride=2, padding=1)
        self.unpool1 = torch.nn.MaxUnpool3d(kernel_size=2, padding=1)
        self.deconv1 = torch.nn.ConvTranspose3d(72, output_channels, 2, stride=2, padding=1)

    def forward(self, x):
        # edcoding
        convoluted1 = self.conv1(x)
        pooled1, indices1 = self.pool1(convoluted1)
        convoluted2 = self.conv2(pooled1)
        pooled2, indices2 = self.pool2(convoluted2)

        # decoding
        unpooled2 = self.unpool2(pooled2, indices2, output_size=convoluted2.size())
        deconvoluted2 = self.deconv2(unpooled2, output_size=pooled1.size())
        unpooled1 = self.unpool1(deconvoluted2, indices1, output_size=convoluted1.size())
        deconvoluted1 = self.deconv1(unpooled1, output_size=x.size())

        return deconvoluted1


class CAEConfig(EstimatorConfig):
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
    def load(cls, path: str) -> 'EstimatorConfig':
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


class CAE(Estimator):

    def __init__(self, config: CAEConfig):
        super().__init__(config)

        self.config = config

        self.model = CAEModel(self.config.n_input_channels,
                                                    self.config.n_output_channels)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.runner = SupervisedRunner()
        self.model_metrics = dict()

    def predict(self, inputs):
        return self.model(torch.as_tensor(inputs)).cpu().data.numpy()

    def metrics(self) -> Dict[str, float]:
        return self.model_metrics

    def train(self, inputs, targets=None):
        plugin = HDF5DatasetPyTorchSplitterPlugin(4)
        loaders = plugin.apply_on_x_y(inputs, targets)

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

        config = CAEConfig.load(params_save_path)

        estimator = CAE(config)
        model = estimator.model.load_state_dict(torch.load(model_save_path))
        # model.eval()
        estimator.model = model
        return estimator
