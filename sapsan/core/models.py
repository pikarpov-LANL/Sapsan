import json
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np


class EstimatorConfig(ABC):
    """ Estimator Config class """
    @classmethod
    @abstractmethod
    def load(cls, path: str):
        """
        Parse yaml file with Config
        @param path: path to yaml Config
        @return: instance of Config
        """
        pass

    @abstractmethod
    def to_dict(self):
        pass

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)


class Estimator(ABC):
    """ Parent class of all estimators """
    def __init__(self, config: EstimatorConfig):
        self.config = config

    @abstractmethod
    def train(self, inputs, targets=None):
        """
        Trains estimator based on inputs and targets
        @param inputs:
        @param targets:
        @return:
        """
        pass

    @abstractmethod
    def predict(self, inputs):
        """
        Make a prediction based on inputs and state of estimator
        @param inputs:
        @return:
        """
        pass

    @abstractmethod
    def metrics(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def save(self, path):
        pass


class Dataset(ABC):
    """ Abstract class for sapsan dataset loader """
    @abstractmethod
    def load(self):
        """
        Loads dataset
        @return:
        """
        pass


class ExperimentBackend(ABC):
    """ Backend of experiment. """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def log_metric(self, name: str, value: float):
        pass

    @abstractmethod
    def log_parameter(self, name: str, value: str):
        pass

    @abstractmethod
    def log_artifact(self, path: str):
        pass


class Experiment(ABC):
    """ Abstract class for sapsan experiments """

    def __init__(self,
                 name: str,
                 backend: ExperimentBackend):
        self.name = name
        self.backend = backend

    @abstractmethod
    def run(self) -> dict:
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def get_artifacts(self) -> List[str]:
        pass


class DatasetPlugin(ABC):
    """ Plugin for dataset.
    Example: convert dataset x, y to pytorch loaders
    """
    @abstractmethod
    def apply(self, dataset: Dataset):
        pass

    @abstractmethod
    def apply_on_x_y(self, x: np.ndarray, y: np.ndarray):
        pass


class Callback(ABC):
    """ Utility class for callbacks implementations. """
    @abstractmethod
    def call(self):
        pass


class Sampling(ABC):
    @abstractmethod
    def sample(self, data: np.ndarray):
        pass

    @property
    @abstractmethod
    def sample_dim(self):
        pass


class Shaper(ABC):
    @abstractmethod
    def apply(self, input: np.ndarray) -> np.ndarray:
        pass
