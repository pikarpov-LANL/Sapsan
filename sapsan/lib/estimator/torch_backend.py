"""
Backend for pytorch-based models

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

import torch
from catalyst.dl import SupervisedRunner, EarlyStoppingCallback, CheckpointCallback, SchedulerCallback, DeviceEngine
from sapsan.core.models import Estimator, EstimatorConfig

class SkipCheckpointCallback(CheckpointCallback):
    def on_epoch_end(self, state):
        pass
    
class TorchBackend(Estimator):
    def __init__(self, config: EstimatorConfig, model):
        super().__init__(config)

        self.runner = SupervisedRunner()
        self.model_metrics = dict()
        self.model = model
        self.ddp = False
        self.set_device()
           
    def predict(self, inputs, config):
        self.model.eval()
        
        #overwrite device and ddp setting if provided upon loading the model,
        #otherwise device will be determined by availability and ddp=False
        if 'device' in config.kwargs: self.device = config.kwargs['device']
        if 'ddp' in config.kwargs: self.ddp = config.kwargs['ddp']
        
        self.print_info()
        
        if str(self.device) == 'cpu' or self.ddp==True: 
            data = torch.as_tensor(inputs)            
        else: 
            if not next(self.model.parameters()).is_cuda: self.model.to(self.device)
            data = torch.as_tensor(inputs).cuda()

        return self.model(data).cpu().data.numpy()               

    def metrics(self) -> Dict[str, float]:
        return self.model_metrics
    
    def set_device(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        return self.device
    
    def print_info(self):
        return print('''
 ===== run info =====
 Device used: {device}
 DDP: {ddp}
 ====================
 '''.format(device=self.device, ddp=self.ddp))
    
    def to_device(self, var):
        if str(self.device) == 'cpu': return var
        else: return var.cuda()
        
    def tensor_to_device(self):
        if str(self.device) == 'cpu': return torch.FloatTensor
        else: return torch.cuda.FloatTensor 
    
    def import_from_config(self):
        if self.config.kwargs:
            for key, value in self.config.kwargs.items():
                setattr(self, key, value)
        return
        
    def torch_train(self, loaders, model, 
                    optimizer, loss_func, scheduler, 
                    config):
        self.config = config
        self.model = model        
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.scheduler = scheduler
        self.loader_key = list(loaders)[0]
        self.metric_key = 'loss'        
        self.import_from_config()

        self.print_info()
        
        ##checks if logdir exists - deletes it if yes
        self.check_logdir()               
        
        if self.loader_key != 'train': 
            warnings.warn("WARNING: loader to be used for early-stop callback is '%s'. You can define it manually in /lib/estimator/pytorch_estimator.torch_train"%(self.loader_key))

        model = self.model        
        
        torch.cuda.empty_cache()
            
        self.runner.train(model=model,
                          criterion=self.loss_func,
                          optimizer=self.optimizer,
                          scheduler=self.scheduler,
                          loaders=loaders,
                          logdir=self.config.logdir,
                          num_epochs=self.config.n_epochs,
                          callbacks=[EarlyStoppingCallback(patience=self.config.patience,
                                                           min_delta=self.config.min_delta,
														   loader_key=self.loader_key,
														   metric_key=self.metric_key,
														   minimize=True),
                                    SchedulerCallback(loader_key=self.loader_key,
                                                      metric_key=self.metric_key,),
                                    SkipCheckpointCallback(logdir=self.config.logdir)
                                    ],
                          verbose=False,
                          check=False,
                          engine=DeviceEngine(self.device),
                          ddp=self.ddp
                          )
        
        self.config.parameters['model - device'] = str(self.runner.device)
        self.model_metrics['final epoch'] = self.runner.stage_epoch_step
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
    def load(cls, path: str, estimator=None):
        model_save_path = "{path}/model".format(path=path)
        params_save_path = "{path}/params.json".format(path=path)
        
        cfg = cls.load_config(params_save_path)
        
        #only overwrite device and ddp setting if provided when loading the model
        for key, value in cfg.items():
            if key == 'kwargs':
                for k, v in cfg['kwargs'].items():
                    if k=='device' or k=='ddp': pass
                    else: setattr(estimator.config.kwargs, k, v)
            else: setattr(estimator.config, key, value)
            
        estimator.model.load_state_dict(torch.load(model_save_path))
        return estimator
    
    @classmethod
    def load_config(cls, path: str):
        with open(path, 'r') as f:
            cfg = json.load(f)
            del cfg['parameters']
            return cfg
    
    def check_logdir(self):
        #checks if logdir exists - deletes if yes
        if os.path.exists(self.config.logdir):
            shutil.rmtree(self.config.logdir)
  

class load_estimator(TorchBackend):
    def __init__(self, config, 
                       model):
        super().__init__(config, model)

    def train(self): pass
