import logging

from sapsan.core.abstrctions.algorithm import Metric, Artifact, Parameter
from sapsan.core.abstrctions.tracking import TrackingBackend


class LoggingBackend(TrackingBackend):
    def __init__(self):
        self.logger = logging

    def log_parameter(self, parameter: Parameter):
        self.logger.info(parameter)

    def log_artifact(self, artifact: Artifact):
        self.logger.info(artifact)

    def log_metric(self, metric: Metric):
        self.logger.info(metric)
