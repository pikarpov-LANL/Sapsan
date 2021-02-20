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
                 alpha: Optional[float] = None,
                 gamma: Optional[float] = None):
        self.alpha = alpha
        self.gamma = gamma

    @classmethod
    def load(cls, path: Optional[str] = None) -> 'KRRConfig':
        with open(path, 'r') as f:
            cfg = json.load(f)
            return cls(**cfg)

    def to_dict(self):
        return {
            "alpha": self.alpha,
            "gamma": self.gamma
        }


class KRR(Estimator):
    _SAVE_PATH_MODEL_SUFFIX = "model"
    _SAVE_PATH_PARAMS_SUFFIX = "params"

    def __init__(self, config: KRRConfig):
        super().__init__(config)

        self.config = config

        self.model = KernelRidge(kernel='rbf')
        if config.gamma and config.alpha:
            self.model = KernelRidge(kernel='rbf', alpha=config.alpha, gamma=config.gamma)
        self.model_metrics = dict()

    def _move_axis_to_sklearn(self, inputs: np.ndarray) -> np.ndarray:
        return np.moveaxis(inputs, 0, 1)

    def predict(self, inputs):
        return self.model.predict(self._move_axis_to_sklearn(inputs))

    def train(self, inputs, targets=None):
        model = self.model.fit(self._move_axis_to_sklearn(inputs), self._move_axis_to_sklearn(targets))
        self.model = model

    def metrics(self) -> Dict[str, float]:
        return self.model_metrics

    def save(self, path):
        model_save_path = "{path}/{suffix}.json".format(path=path, suffix=self._SAVE_PATH_MODEL_SUFFIX)
        params_save_path = "{path}/{suffix}.json".format(path=path, suffix=self._SAVE_PATH_PARAMS_SUFFIX)

        dump(self.model, model_save_path)
        self.config.save(params_save_path)

    @classmethod
    def load(cls, path):
        model_save_path = "{path}/{suffix}.json".format(path=path, suffix=cls._SAVE_PATH_MODEL_SUFFIX)
        params_save_path = "{path}/{suffix}.json".format(path=path, suffix=cls._SAVE_PATH_PARAMS_SUFFIX)

        model = load(model_save_path)
        config = KRRConfig.load(params_save_path)
        estimator = KRR(config)
        estimator.model = model
        return estimator
