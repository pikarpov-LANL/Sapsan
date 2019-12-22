from sapsan.general.models import EstimatorConfiguration, Estimator


class KrrEstimarotConfiguration(EstimatorConfiguration):
    def __init__(self,
                 alpha: float,
                 gamma: float):
        self.alpha = alpha
        self.gamma = gamma

    @classmethod
    def from_yaml(cls, path: str) -> 'EstimatorConfiguration':
        pass


class KrrEstimator(Estimator):
    def __init__(self, config: KrrEstimarotConfiguration):
        super().__init__(config)

        self.config = config

    def predict(self, inputs):
        pass

    def train(self, inputs, targets=None):
        pass
