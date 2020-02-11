import time
from typing import List, Dict

import mlflow
import numpy as np
from sapsan.general.models import Experiment, ExperimentBackend, Estimator


class MlFlowExperimentBackend(ExperimentBackend):
    def __init__(self,
                 name: str,
                 host: str,
                 port: int):
        self.name = name
        self.host = host
        self.port = port
        self.mlflow_url = "http://{host}:{port}".format(host=host,
                                                        port=port)
        mlflow.set_tracking_uri(self.mlflow_url)
        self.experiment_id = mlflow.set_experiment(name)

    def log_metric(self, name: str, value: float):
        mlflow.log_metric(name, value)

    def log_parameter(self, name: str, value: str):
        mlflow.log_param(name, value)

    def log_artifact(self, path: str):
        # TODO
        pass


class TrainingExperiment(Experiment):
    def __init__(self,
                 name: str,
                 backend: ExperimentBackend,
                 model: Estimator,
                 inputs: np.ndarray,
                 targets: np.ndarray):
        super().__init__(name, backend)
        self.model = model
        self.inputs = inputs
        self.targets = targets

    def get_metrics(self) -> Dict[str, float]:
        return self.model.metrics()

    def get_parameter(self) -> Dict[str, str]:
        return self.model.config.to_dict()

    def get_artifacts(self) -> List[str]:
        pass

    def run(self):
        start = time.time()
        self.model.train(self.inputs, self.targets)
        end = time.time()

        runtime = end - start

        for metric, value in self.get_metrics().items():
            self.backend.log_metric(metric, value)

        for param, value in self.get_parameter().items():
            self.backend.log_parameter(param, value)

        return {
            'runtime': runtime
        }

