"""
Backend for sklearn-based models

    - configuring to run either on cpu or gpu
    - loading parameters into a catalyst runner
    - output the metrics and model details 
"""
import json
from typing import Dict
import numpy as np
import warnings
import os
import shutil
from joblib import dump, load

from sapsan.core.models import Estimator, EstimatorConfig

class SklearnBackend(Estimator):
    def __init__(self, config: EstimatorConfig, model):
        super().__init__(config)
        
        self.model_metrics = dict()
        self.model = model
        
    def predict(self, inputs, config):
        pred = self.model.predict(self._move_axis_to_sklearn(inputs))
        self.model_metrics['eval - R2'] = self.model.score(self._move_axis_to_sklearn(inputs), pred)
        return pred  
    
    def save(self, path):
        model_save_path = "{path}/model.json".format(path=path)
        params_save_path = "{path}/params.json".format(path=path)

        dump(self.model, model_save_path)
        self.config.save(params_save_path)
        
    @classmethod
    def load(cls, path: str, estimator = None):
        model_save_path = "{path}/model.json".format(path=path)
        params_save_path = "{path}/params.json".format(path=path)
                
        cfg = cls.load_config(params_save_path)
        for key, value in cfg.items():
            setattr(estimator.config, key, value)
        
        estimator.model = load(model_save_path)
        return estimator
    
    @classmethod
    def load_config(cls, path: str):
        with open(path, 'r') as f:
            cfg = json.load(f)
            del cfg['parameters']
            return cfg
        
class load_sklearn_estimator(SklearnBackend):
    def __init__(self, config, 
                       model):
        super().__init__(config, model)

    def train(self): pass