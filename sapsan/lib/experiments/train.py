import time
from typing import Dict, List

import numpy as np
from sapsan.core.models import Experiment, ExperimentBackend, Estimator
from sapsan.lib.backends.fake import FakeBackend
from sapsan.utils.plot import log_plot

import os
import sys
import inspect

class Train(Experiment):

    def __init__(self,
                 model: Estimator,
                 data_parameters,
                 backend = FakeBackend(),
                 show_log: bool = True,
                 run_name: str = 'train'
                ):
        self.backend = backend
        self.model = model
        self.data_parameters = data_parameters
        self.artifacts = []
        self.show_log = show_log
        self.run_name = run_name

    def get_metrics(self) -> Dict[str, float]:
        return self.model.metrics()

    def get_parameters(self) -> Dict[str, str]:
        return {**self.data_parameters.get_parameters(), **self.model.config.parameters}

    def get_artifacts(self) -> List[str]:
        return self.artifacts

    def _cleanup(self):
        for artifact in self.artifacts:
            os.remove(artifact)
        return len(self.artifacts)
    
    def run(self):
        
        start = time.time() 
        self.backend.close_active_run()
        self.run_id = self.backend.start(self.run_name)
        
        self.model.train() 
        
        end = time.time()
        runtime = end - start
        
        #only if catalyst.runner is used
        if os.path.exists('model_details.txt'):
            self.artifacts.append('model_details.txt')                        
            
            #plot the training log if pytorch is used
            try: 
                log = log_plot(self.show_log)
                log.write_html("runtime_log.html")
                self.artifacts.append("runtime_log.html")
            except: print("WARNING: Couldn't plot the training log - train.csv not found")
        else: pass
        
        try:
            with open('model_forward.txt', 'w') as fw:
                fw.write(inspect.getsource(self.model.model.forward))
            self.artifacts.append('model_forward.txt')
        except: pass
        
        #only if catalyst.runner is used
        if 'train' in self.get_metrics():
            self.backend.log_metric( 'train - final epoch', self.get_metrics()['final epoch'])        
            for metric, value in self.get_metrics()['train'].items():
                if "/" in metric: metric = metric.replace("/", " over ")
                self.backend.log_metric('train - %s'%metric, value)            

        for param, value in self.get_parameters().items():
            self.backend.log_parameter(param, value)
            
        for artifact in self.get_artifacts():
            self.backend.log_artifact(artifact)

        self.backend.log_metric("train - runtime", runtime)
        
        #attempts to remove all artifacts
        try: self._cleanup()
        except: pass

        print('runtime %.4f seconds'%runtime)
        
        return self.model
