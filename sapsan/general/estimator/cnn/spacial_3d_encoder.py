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
                      checkpoints=[0])

    x, y = ds.load()

    model = estimator.train(x, y)
"""

import torch
from catalyst.dl import SupervisedRunner, EarlyStoppingCallback

from sapsan.general.data.jhtdb_dataset import JHTDBDatasetPyTorchSplitterPlugin, JHTDBDataset
from sapsan.general.models import Estimator, EstimatorConfiguration


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
                 n_epochs: int):
        self.n_epochs = n_epochs

    @classmethod
    def from_yaml(cls, path: str) -> 'EstimatorConfiguration':
        pass


class Spacial3dEncoderNetworkEstimator(Estimator):
    def __init__(self, config: Spacial3dEncoderNetworkEstimatorConfiguration):
        super().__init__(config)

        self.config = config

        # todo: unhardcode
        self.model = Spacial3dEncoderNetworkModel(36, 64**3*3)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.logdir = "./logs/"

    def predict(self, inputs):
        pass

    def train(self, inputs, targets=None):
        plugin = JHTDBDatasetPyTorchSplitterPlugin(4)
        loaders = plugin.apply_on_x_y(inputs, targets)

        model = self.model
        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_func = torch.nn.MSELoss()  # torch.nn.SmoothL1Loss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=3,
            min_lr=1e-5)

        runner = SupervisedRunner()

        runner.train(model=model,
                     criterion=loss_func,
                     optimizer=optimizer,
                     scheduler=scheduler,
                     loaders=loaders,
                     logdir=self.logdir,
                     num_epochs=self.config.n_epochs,
                     callbacks=[EarlyStoppingCallback(patience=10, min_delta=1e-5)],
                     verbose=False,
                     check=False)

        return model
