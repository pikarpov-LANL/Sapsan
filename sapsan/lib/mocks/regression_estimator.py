from sklearn.linear_model import LinearRegression

from sapsan.core.models import Estimator, EstimatorConfiguration


class LinearRegressionEstimatorConfiguration(EstimatorConfiguration):

    def __init__(self, seed: int = 42):
        self.seed = seed

    @classmethod
    def from_yaml(cls, path: str) -> 'EstimatorConfiguration':
        pass


class LinearRegressionEstimator(Estimator):
    def __init__(self,
                 config: LinearRegressionEstimatorConfiguration = LinearRegressionEstimatorConfiguration()):
        super().__init__(config)
        self.config = config
        self.model = LinearRegression()

    def train(self, inputs, targets=None):
        self.model.fit(inputs, targets)
        return self.model

    def predict(self, inputs):
        return self.model.predict(inputs)
