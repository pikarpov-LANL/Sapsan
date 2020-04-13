from abc import ABC, abstractmethod
from typing import Union, List, Optional

import numpy as np

from sapsan.core.abstrctions.dataset import Dataset


class Metric:
    """
    Metric
    """
    def __init__(self,
                 key: str,
                 value: Union[int, float]):
        """

        @param key: key
        @param value: value
        """
        self.key = key
        self.value = value

    def __repr__(self):
        return "Metric[{0}:{1}]".format(self.key, self.value)


class Parameter:
    """
    Parameter
    """
    def __init__(self,
                 key: str,
                 value: Union[int, float, str]):
        """

        @param key: key
        @param value: value
        """
        self.key = key
        self.value = value

    def __repr__(self):
        return "Parameter[{0}:{1}]".format(self.key, self.value)


class Artifact:
    """
    Artifact
    """
    def __init__(self,
                 name: str,
                 path: str):
        """

        @param name: name of artifact
        @param path: path to artifact
        """
        self.name = name
        self.path = path

    def __repr__(self):
        return "Artifact[{0}:{1}]".format(self.name, self.path)


class Algorithm(ABC):
    """
    Base class for all algorithms.
    """
    @abstractmethod
    def save(self, path: str):
        """
        Saving algorithm
        @param path: path to saved file
        @return:
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str):
        """
        Loading algorithm
        @param path: path to pickled files
        @return:
        """
        pass


class Estimator(Algorithm):
    """
    Base class for estimator type of algorithms
    """
    @abstractmethod
    def train(self,
              features: Union[np.array, Dataset],
              targets: Optional[Union[np.array, Dataset]] = None):
        """
        Training of algorithm
        @param features: features to train on
        @param targets: labels to fit to
        @return:
        """
        pass

    @abstractmethod
    def predict(self,
                data: Union[np.array, Dataset]) -> Union[np.array, Dataset]:
        """
        Prediction
        @param data: input data
        @return: prediction result
        """
        pass


class BaseAlgorithm(Algorithm):
    @abstractmethod
    def run(self,
            data: Optional[Union[np.array, Dataset]] = None):
        """
        Run of algorithm
        @param data: input data
        @return:
        """
        pass
