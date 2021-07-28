import json
import os

import yaml

import numpy as np
from typing import Optional, Dict
from sklearn.kernel_ridge import KernelRidge
from sapsan.core.models import Estimator, EstimatorConfig
from joblib import dump, load

class KRRConfig(EstimatorConfig):
    def __init__(self,
                 alpha: float = 1.0,
                 gamma: float = 1.0,
                 kernel: str = 'rbf',
                 *args, **kwargs):
        self.alpha = alpha
        self.gamma = gamma
        self.kernel = kernel
        self.kwargs = kwargs
        
        #everything in self.parameters will get recorded by MLflow
        #in the case of scikit-learn (KRR), parameters are populated in __init__ of KRR class
        self.parameters = {}
        

class KRR(Estimator):

    def __init__(self, config: KRRConfig):
        super().__init__(config)

        self.config = config

        self.model = KernelRidge(kernel='rbf')
        if config.gamma and config.alpha:
            self.model = KernelRidge(kernel=config.kernel, alpha=config.alpha, gamma=config.gamma)
        
        for param, value in self.model.get_params().items():
            self.config.parameters["model - %s"%param] = value
        self.model_metrics = dict()

    def _move_axis_to_sklearn(self, inputs: np.ndarray) -> np.ndarray:
        return np.moveaxis(inputs, 0, 1)

    def predict(self, inputs):
        pred = self.model.predict(self._move_axis_to_sklearn(inputs))
        self.model_metrics['eval - R2'] = self.model.score(self._move_axis_to_sklearn(inputs), pred)
        return pred

    def train(self, loaders):
        model = self.model.fit(self._move_axis_to_sklearn(loaders[0]),
                               self._move_axis_to_sklearn(loaders[1]))
        self.model = model
        return model

    def metrics(self) -> Dict[str, float]:
        return self.model_metrics

    def save(self, path):
        model_save_path = "{path}/model.json".format(path=path)
        params_save_path = "{path}/params.json".format(path=path)

        dump(self.model, model_save_path)
        self.config.save(params_save_path)
        
    @classmethod
    def load(cls, path: str, model=None, config=None):
        model_save_path = "{path}/model.json".format(path=path)
        params_save_path = "{path}/params.json".format(path=path)
        
        config = model.load_config(params_save_path, config)
        
        estimator = model(config)
        model = load(model_save_path)
        estimator.model = model
        return estimator
    
    @classmethod
    def load_config(self, path: str, config = None):
        with open(path, 'r') as f:
            cfg = json.load(f)
            del cfg['parameters']
            return config(**cfg)
