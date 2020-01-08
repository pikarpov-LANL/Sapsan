import yaml
from typing import Optional
from sklearn.kernel_ridge import KernelRidge
from sapsan.general.models import EstimatorConfiguration, Estimator


class KrrEstimatorConfiguration(EstimatorConfiguration):
    def __init__(self,
                 alpha: Optional[float] = None,
                 gamma: Optional[float] = None):
        self.alpha = alpha
        self.gamma = gamma

    @classmethod
    def from_yaml(cls, path: str) -> 'KrrEstimatorConfiguration':
        with open(path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            return cls(**cfg['config'])


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
