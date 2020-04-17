from abc import ABC, abstractmethod
from typing import List

from sapsan.core.abstrctions.algorithm import Parameter, Metric, Artifact
from sapsan.core.abstrctions.tracking import TrackingBackend
from sapsan.core.tracking.logger import LoggingBackend


class Experiment(ABC):
    """
    Base experiment class
    """
    def __init__(self,
                 tracking_backend: TrackingBackend):
        """
        @param tracking_backend: tracking backend
        """
        self.tracking_backend = tracking_backend

    def execute(self, *args, **kwargs):
        result = self.run(*args, **kwargs)
        self.tracking_backend.log_parameters(parameters=self.parameters)
        self.tracking_backend.log_metrics(metrics=self.metrics)
        self.tracking_backend.log_artifacts(artifacts=self.artifacts)
        return result

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Pass of experiment
        @return:
        """
        pass

    @abstractmethod
    def test(self,
             parameters: Parameter):
        """
        Test/evaluation of experiment
        @param parameters: parameters for test
        @return:
        """
        pass

    @property
    @abstractmethod
    def parameters(self) -> List[Parameter]:
        """
        List of parameters of algorithm
        @return: list of parameters for algorithm
        """
        pass

    @property
    @abstractmethod
    def metrics(self) -> List[Metric]:
        """
        List of metrics of algorithm
        @return: list of metrics
        """
        pass

    @property
    @abstractmethod
    def artifacts(self) -> List[Artifact]:
        """
        List of artifacts produced by algorithm
        @return:
        """
        pass