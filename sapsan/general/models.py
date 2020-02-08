from abc import ABC, abstractmethod


class EstimatorConfiguration(ABC):
    """ Estimator configuration class """
    @classmethod
    @abstractmethod
    def from_yaml(cls, path: str) -> 'EstimatorConfiguration':
        """
        Parse yaml file with configuration
        @param path: path to yaml configuration
        @return: instance of configuration
        """
        pass


class Estimator(ABC):
    """ Parent class of all estimators """
    def __init__(self, config: EstimatorConfiguration):
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


class Dataset(ABC):
    """ Abstract class for sapsan dataset loader """
    @abstractmethod
    def load(self):
        """
        Loads dataset
        @return:
        """
        pass


class Experiment(ABC):
    """ Abstract class for sapsan experiments """
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def get_report(self):
        pass


class DatasetPlugin(ABC):
    """ Plugin for dataset.
    Example: convert dataset x, y to pytorch loaders
    """
    @abstractmethod
    def apply(self, dataset: Dataset):
        pass

