from sklearn.metrics import mean_squared_error
from sapsan.core.models import Experiment, Estimator, Dataset


class FakeTrainingExperiment(Experiment):
    def __init__(self,
                 name: str,
                 estimator: Estimator,
                 dataset: Dataset):
        self.estimator = estimator
        self.dataset = dataset
        self.name = name
        self.metrics = {}

    def run(self):
        x, y = self.dataset.load()
        self.estimator.train(x, y)

    def get_report(self):
        return {
            "name": self.name,
            "metrics": self.metrics
        }


class FakeInferenceExperiment(Experiment):
    def __init__(self,
                 name: str,
                 dataset: Dataset,
                 estimator: Estimator):
        self.estimator = estimator
        self.dataset = dataset
        self.name = name
        self.metrics = {}

    def run(self):
        x, y = self.dataset.load()
        predictions = self.estimator.predict(x).reshape(-1)
        mse = mean_squared_error(y.reshape(-1), predictions)
        self.metrics["mse"] = mse
        return predictions

    def get_report(self):
        return {
            "name": self.name,
            "metrics": self.metrics
        }
