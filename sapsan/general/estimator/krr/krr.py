import os

import yaml
from typing import Optional, Dict
from sklearn.kernel_ridge import KernelRidge
from sapsan.general.models import Estimator, EstimatorConfiguration


class KrrEstimatorConfiguration(EstimatorConfiguration):
    def __init__(self,
                 alpha: Optional[float] = None,
                 gamma: Optional[float] = None):
        self.alpha = alpha
        self.gamma = gamma

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> 'KrrEstimatorConfiguration':
        if not path:
            path = "{}/krr_config.yaml".format(
                os.path.dirname(os.path.realpath(__file__))
            )

        with open(path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            return cls(**cfg['config'])

    def to_dict(self):
        return {
            "alpha": self.alpha,
            "gamma": self.gamma
        }


class KrrEstimator(Estimator):
    def __init__(self, config: KrrEstimatorConfiguration):
        super().__init__(config)

        self.config = config

        self.model = KernelRidge(kernel='rbf')
        if config.gamma and config.alpha:
            self.model = KernelRidge(kernel='rbf', alpha=config.alpha, gamma=config.gamma)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def train(self, inputs, targets=None):
        model = self.model.fit(inputs, targets)
        self.model = model

    def metrics(self) -> Dict[str, float]:
        pass
