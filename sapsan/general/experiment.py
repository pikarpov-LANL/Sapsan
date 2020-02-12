"""

Example:
    experiment_name = "Training experiment"
    dataset_root_dir = "/Users/icekhan/Documents/development/myprojects/sapsan/repo/Sapsan/dataset"
    estimator = Spacial3dEncoderNetworkEstimator(
        config=Spacial3dEncoderNetworkEstimatorConfiguration(n_epochs=1)
    )
    x, y = JHTDB128Dataset(path=dataset_root_dir,
                           features=['u', 'b', 'a',
                                'du0', 'du1', 'du2',
                                'db0', 'db1', 'db2',
                                'da0', 'da1', 'da2'],
                           labels=['tn'],
                           checkpoints=[0.0]).load()

    experiment = TrainingExperiment(name=experiment_name,
                                    backend=FakeExperimentBackend(experiment_name),
                                    model=estimator,
                                    inputs=x, targets=y)
    experiment.run()
"""

import time
import logging
from typing import List, Dict

import mlflow
import numpy as np

from sapsan.general.data.jhtdb_dataset import JHTDB128Dataset
from sapsan.general.estimator.cnn.spacial_3d_encoder import Spacial3dEncoderNetworkEstimator, \
    Spacial3dEncoderNetworkEstimatorConfiguration
from sapsan.general.models import Experiment, ExperimentBackend, Estimator


class FakeExperimentBackend(ExperimentBackend):
    def log_parameter(self, name: str, value: str):
        logging.info("Logging experiment '{experiment}' parameter "
                     "{name}: {value}".format(experiment=self.name,
                                              name=name,
                                              value=value))

    def log_artifact(self, path: str):
        logging.info("Logging artifact {path}".format(path=path))

    def log_metric(self, name: str, value: float):
        logging.info("Logging experiment '{experiment}' metric "
                     "{name}: {value}".format(experiment=self.name,
                                              name=name,
                                              value=value))


class MlFlowExperimentBackend(ExperimentBackend):
    def __init__(self, name: str, host: str, port: int):
        super().__init__(name)
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

    def get_parameters(self) -> Dict[str, str]:
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

        for param, value in self.get_parameters().items():
            self.backend.log_parameter(param, value)

        self.backend.log_metric("runtime", runtime)

        return {
            'runtime': runtime
        }
