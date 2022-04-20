"""
Backend for pytorch-based models

    - configuring to run either on cpu or gpu
    - loading parameters into a catalyst runner
    - output the metrics and model details 
    - saving and loading trained models
    - predicting
    - customize Catalyst Runner
        - set self.runner in TorchBackend to the one you like or custom
    - setup a custom Distributed Data Parallel (DDP) run
        - adjust loaders in TorchBackend.torch_train()
        - adjust self.runner.train() settings
"""
import json
from typing import Dict
import numpy as np
import warnings
import os
import shutil

import torch
from catalyst.dl import SupervisedRunner, EarlyStoppingCallback, CheckpointCallback, SchedulerCallback, DeviceEngine, DistributedDataParallelEngine
from catalyst import dl
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
        
        self.set_device()

        if 'cuda' in str(self.device):
            self.optimizer_to(optimizer, self.device)
        
        #checks if logdir exists - deletes it if yes
        self.check_logdir()               
        
        if self.loader_key != 'train': 
            warnings.warn("WARNING: loader to be used for early-stop callback is '%s'. You can define it manually in /lib/estimator/pytorch_estimator.torch_train"%(self.loader_key))

        model = self.model        
        
        torch.cuda.empty_cache()        

        if self.ddp: self.engine = None
        else: self.engine = DeviceEngine(self.device)
                                
        self.print_info()
        
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
                                    SkipCheckpointCallback(logdir=self.config.logdir),
                                    ],
                          verbose=False,
                          check=False,
                          engine=self.engine,
                          ddp=self.ddp,
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
        
        
    def predict(self, inputs, config):
        self.model.eval()
        
        #overwrite device and ddp setting if provided upon loading the model,
        #otherwise device will be determined by availability and ddp=False
        if 'device' in config.kwargs: self.device = config.kwargs['device']
        if 'ddp' in config.kwargs: self.ddp = config.kwargs['ddp']
        
        self.print_info()
        
        if str(self.device) == 'cpu':
            data = torch.as_tensor(inputs)            
        else: 
            if not next(self.model.parameters()).is_cuda: self.model.to(self.device)
            cuda_id = next(self.model.parameters()).get_device()
            data = torch.as_tensor(inputs).cuda(cuda_id)

        return self.model(data).cpu().data.numpy()               

    def metrics(self) -> Dict[str, float]:
        return self.model_metrics
    
    def set_device(self):
        if hasattr(self.config, 'kwargs') and 'device' in self.config.kwargs: 
            self.device = torch.device(self.config.kwargs['device'])
        else: 
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        return self.device
    
    def print_info(self):
        if self.ddp and torch.cuda.is_available(): device_to_print = 'parallel cuda'
        else: device_to_print = self.device
        
        return print('''
 ====== run info ======
 Device used:  {device}
 DDP:          {ddp}
 ======================
 '''.format(device=device_to_print, ddp=self.ddp))         
    
    def import_from_config(self):
        if self.config.kwargs:
            for key, value in self.config.kwargs.items():
                setattr(self, key, value)
        return
        

    def optimizer_to(self, optim, device):
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)    
    
    def save(self, path):
        model_save_path = "{path}/model.pt".format(path=path)
        params_save_path = "{path}/params.json".format(path=path)
        
        torch.save({
                    'epoch': self.runner.stage_epoch_step,
                    'model_state_dict': self.runner.model.state_dict(),
                    'optimizer_state_dict': self.runner.optimizer.state_dict(),
                    'loss': self.runner.epoch_metrics['train']['loss'],
                    }, model_save_path)
        self.config.save(params_save_path)

    @classmethod
    def load(cls, path: str, estimator=None, load_saved_config=False):
        model_save_path = "{path}/model.pt".format(path=path)
        params_save_path = "{path}/params.json".format(path=path)
        
        cfg = cls.load_config(params_save_path)

        if load_saved_config==True: 
            print('''All config parameters will be loaded from saved params.json 
(anything provided in model config upon loading will be ignored)''')
            for key, value in cfg.items():
                setattr(estimator.config, key, value)

        checkpoint = torch.load(model_save_path, map_location='cpu')
        estimator.model.load_state_dict(checkpoint['model_state_dict'])
        estimator.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        print('''
 ==== Loaded Model ====
 final Epoch: {epoch}
 final Loss: {loss}
 ======================

'''.format(epoch=epoch, loss='%.4e'%loss) )
        
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