import time
from typing import Dict, List

import numpy as np
from sapsan.core.models import Experiment, ExperimentBackend, Estimator


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

