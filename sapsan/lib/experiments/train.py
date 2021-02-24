import time
from typing import Dict, List

import numpy as np
from sapsan.core.models import Experiment, ExperimentBackend, Estimator
from sapsan.utils.plot import log_plot
from sapsan.lib.backends.fake import FakeBackend

import os
import sys

class Train(Experiment):

    def __init__(self,
                 backend: ExperimentBackend,
                 model: Estimator,
                 inputs: np.ndarray,
                 targets: np.ndarray,
                 data_parameters: dict,
                 show_history = True,
                ):
        super().__init__(backend.name, backend)
        self.model = model
        self.inputs = inputs
        self.targets = targets
        self.data_parameters = data_parameters
        self.artifacts = []
        self.show_history = show_history

    def get_metrics(self) -> Dict[str, float]:
        return self.model.metrics()

    def get_parameters(self) -> Dict[str, str]:
        return {**self.data_parameters, **self.model.config.to_dict()}

    def get_artifacts(self) -> List[str]:
        return self.artifacts

    def _cleanup(self):
        for artifact in self.artifacts:
            os.remove(artifact)
        return len(self.artifacts)
    
    def run(self):
        start = time.time()
        
        self.backend.start('train')
                
        self.model.train(self.inputs, self.targets)
        end = time.time()
        runtime = end - start
        
        #if catalyst.runner is not used, then this file won't exist
        if os.path.exists('model_details.txt'):
            self.artifacts.append('model_details.txt')
            
            #plot the training history if pytorch is used
            log = log_plot(self.show_history)
            log.write_html("runtime_log.html")
            self.artifacts.append("runtime_log.html")
        else: pass
        


        for metric, value in self.get_metrics().items():
            self.backend.log_metric(metric, value)

        for param, value in self.get_parameters().items():
            self.backend.log_parameter(param, value)
            
        for artifact in self.get_artifacts():
            self.backend.log_artifact(artifact)

        self.backend.log_metric("runtime", runtime)
        
        self.backend.end()
        self._cleanup()

        return {
            'runtime': runtime
        }
