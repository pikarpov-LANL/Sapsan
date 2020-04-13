from abc import ABC, abstractmethod
from typing import List

from sapsan.core.abstrctions.algorithm import Metric, Parameter, Artifact


class TrackingBackend(ABC):
    def log_metrics(self, metrics: List[Metric]):
        """
        Metrics logging
        @param metrics: metric to log
        @return:
        """
        for metric in metrics:
            self.log_metric(metric)

    def log_parameters(self, parameters: List[Parameter]):
        """
        Parameters logging
        @param parameters: parameters to log
        @return:
        """
        for parameter in parameters:
            self.log_parameter(parameter)

    def log_artifacts(self, artifacts: List[Artifact]):
        """
        Artifacts logging/saving
        @param artifacts: artifacts to log
        @return:
        """
        for artifact in artifacts:
            self.log_artifact(artifact)

    @abstractmethod
    def log_metric(self, metric: Metric):
        """
        Metric logging
        @param metric: metric to log
        @return:
        """
        pass

    @abstractmethod
    def log_parameter(self, parameter: Parameter):
        """
        Parameter logging
        @param parameter: parameter to log
        @return:
        """
        pass

    @abstractmethod
    def log_artifact(self, artifact: Artifact):
        """
        Artifact logging/saving
        @param artifact: artifact to log
        @return:
        """
        pass
