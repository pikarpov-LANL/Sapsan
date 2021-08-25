import json
import os

import numpy as np
from typing import Optional, Dict
from sklearn.kernel_ridge import KernelRidge
from sapsan.core.models import Estimator, EstimatorConfig
from sapsan.lib.estimator.sklearn_backend import SklearnBackend

class KRRModel():
    def __init__(self, kernel='rbf', alpha=1.0, gamma=None):
        super(KRRModel, self).__init__()
        self.kernel = kernel
        self.alpha = alpha
        self.gamma = gamma
        self.model = KernelRidge(kernel = self.kernel,
                                 alpha = self.alpha,
                                 gamma = self.gamma)
    
class KRRConfig(EstimatorConfig):
    def __init__(self,
                 alpha: float = 1.0,
                 gamma = False,
                 kernel: str = 'rbf',
                 *args, **kwargs):
        self.alpha = alpha
        self.gamma = gamma
        self.kernel = kernel
        self.kwargs = kwargs
        
        #everything in self.parameters will get recorded by MLflow
        #in the case of scikit-learn (KRR), parameters are populated in __init__ of KRR class
        self.parameters = {f'model - {k}': v for k, v in self.__dict__.items() if k != 'kwargs'}
        if bool(self.kwargs): self.parameters.update({f'model - {k}': v for k, v in self.kwargs.items()})
        

class KRR(SklearnBackend):
    def __init__(self, loaders,
                       config = KRRConfig(),
                       model = KRRModel()):
        super().__init__(config, model)
        self.config = config
        self.loaders = loaders
        
        self.estimator = KRRModel(kernel=config.kernel, alpha=config.alpha, gamma=config.gamma)
        self.model = self.estimator.model
        
        for param, value in self.model.get_params().items():
            self.config.parameters["model - %s"%param] = value
        self.model_metrics = dict()

    def train(self):
        trained_model = self.model.fit(self._move_axis_to_sklearn(self.loaders[0]),
                                       self._move_axis_to_sklearn(self.loaders[1]))
        return trained_model        
        
    def _move_axis_to_sklearn(self, inputs: np.ndarray) -> np.ndarray:
        return np.moveaxis(inputs, 0, 1)    

    def metrics(self) -> Dict[str, float]:
        return self.model_metrics
