"""
Backend for pytorch-based models

    - configuring to run either on cpu or gpu
    - loading parameters into a catalyst runner
    - output the metrics and model details 
"""
import json
from typing import Dict
import numpy as np

import torch
from catalyst.dl import SupervisedRunner, EarlyStoppingCallback, CheckpointCallback, IterationCheckpointCallback

from sapsan.lib.data.hdf5_dataset import HDF5DatasetPyTorchSplitterPlugin, OutputFlatterDatasetPlugin
from sapsan.core.models import Estimator, EstimatorConfig

class SkipCheckpointCallback(CheckpointCallback):
    def on_epoch_end(self, state):
        pass
    
class TorchEstimator(Estimator):
    def __init__(self, config: EstimatorConfig, model):
        super().__init__(config)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.runner = SupervisedRunner()
        self.model_metrics = dict()
        self.model = model
           
    def predict(self, inputs):
        if str(self.device) == 'cpu': data = torch.as_tensor(inputs)
        else: data = torch.as_tensor(inputs).cuda()
        
        return self.model(data).cpu().data.numpy()

    def metrics(self) -> Dict[str, float]:
        return self.model_metrics
        
    def torch_train(self, inputs, targets, model, optimizer, loss_func, scheduler, config):
        self.config = config
        self.model = model        
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.scheduler = scheduler
        
        print('Device used:', self.device)
                
        output_flatter = OutputFlatterDatasetPlugin()
        splitter_pytorch = HDF5DatasetPyTorchSplitterPlugin(4)
        _, flatten_targets = output_flatter.apply_on_x_y(inputs, targets)
        loaders = splitter_pytorch.apply_on_x_y(inputs, flatten_targets)

        model = self.model
        if torch.cuda.device_count() > 1:
            print("GPUs available: ", torch.cuda.device_count())
            print("Note: if batch_size == 1, then only 1 GPU will be used")
            model = torch.nn.DataParallel(model)
        
        model.to(self.device)

        torch.cuda.empty_cache()
        self.runner.train(model=model,
                          criterion=self.loss_func,
                          optimizer=self.optimizer,
                          scheduler=self.scheduler,
                          loaders=loaders,
                          logdir=self.config.logdir,
                          num_epochs=self.config.n_epochs,
                          callbacks=[EarlyStoppingCallback(patience=self.config.patience,
                                                           min_delta=self.config.min_delta),
                                    SkipCheckpointCallback()
                                    ],
                          verbose=False,
                          check=False)
        
        self.config.parameters['model - device'] = self.runner.device         
        self.model_metrics['final epoch'] = self.runner.epoch-1
        for key,value in self.runner.epoch_metrics.items():
            self.model_metrics[key] = value

        with open('model_details.txt', 'w') as file:
            file.write('%s\n\n%s\n\n%s'%(str(self.runner.model),
                                   str(self.runner.optimizer),
                                   str(self.runner.scheduler)))
        
        return model

    def save(self, path):
        model_save_path = "{path}/model".format(path=path)
        params_save_path = "{path}/params.json".format(path=path)

        torch.save(self.model.state_dict(), model_save_path)
        self.config.save(params_save_path)

    @classmethod
    def load(cls, path: str, model=None, config=None):
        model_save_path = "{path}/model".format(path=path)
        params_save_path = "{path}/params.json".format(path=path)

        config = config.load(params_save_path)

        estimator = model(config)
        model = estimator.model.load_state_dict(torch.load(model_save_path))
        # model.eval()
        estimator.model = model
        return estimator