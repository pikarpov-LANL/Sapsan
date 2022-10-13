"""
Physics-Informed Convolutional model to predict
the diagonal Reynolds stress tensor terms, which
can later be used to calculate Turbulent Pressure.

The loss function here is a custom combination of
statistical (Kolmogorov-Smirnov) and spatial (Smooth L1)
losses.

The model has been published in P.I.Karpov, arXiv:2205.08663

"""

import json
import numpy as np
import sys
import os

import torch
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sapsan.core.models import EstimatorConfig
from sapsan.lib.estimator.torch_backend import TorchBackend
from sapsan.lib.data import get_loader_shape

from sapsan.lib.estimator.torch_modules import Interp1d
from sapsan.lib.estimator.torch_modules import Gaussian

class PIMLTurbModel(torch.nn.ModuleDict):
    def __init__(self, D_in = 1, D_out = 1, activ = "ReLU", sigma=1):
        super(PIMLTurbModel, self).__init__()       
        
        if D_out>=116**3: stride1=2; stride2=2; stride3=2
        elif D_out>=39**3: stride1=1; stride2=2; stride3=2
        elif D_out>=17**3: stride1=1; stride2=1; stride3=1
        else: stride1=1; stride2=1; stride3=1
            
        self.conv3d1 = torch.nn.Conv3d(D_in, D_in*2, kernel_size=2, stride=stride1)
        self.conv3d2 = torch.nn.Conv3d(D_in*2, D_in*4, kernel_size=2, stride=stride2)
        self.conv3d3 = torch.nn.Conv3d(D_in*4, D_in*8, kernel_size=2, stride=stride3)

        self.pool = torch.nn.MaxPool3d(kernel_size=2)

        self.activation = getattr(torch.nn, activ)()

        self.linear = torch.nn.Linear(D_in*8, D_in*16)
        self.linear2 = torch.nn.Linear(D_in*16, D_out)
        
        self.gaussian = Gaussian(sigma=sigma)

    def forward(self, x):         
        x = x.float()
        x_shape = x.size()
        
        x = self.conv3d1(x)        
        x = self.pool(x)
        
        x = self.activation(x)        
        x = self.conv3d2(x)
        x = self.pool(x)
        
        x = self.activation(x)        
        x = self.conv3d3(x)
        x = self.pool(x)

        x = self.activation(x) 
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.activation(x)
        x = self.linear2(x) 
        
        x_shape = list(x_shape)
        x_shape[1]=1        
        x = torch.reshape(x,x_shape)                                       
        x = torch.mul(x,x)
        x = self.gaussian(x)
        
        return x  

class SmoothL1_KSLoss(torch.nn.Module):
    '''
    The loss functions combines a statistical (Kolmogorov-Smirnov)
    and a spatial loss (Smooth L1).
    
    Corresponding 'train_l1ks_log.txt' and 'valid_l1ks_log.txt'
    are written out to include the individual loss evolutions.
    '''    
    def __init__(self, ks_stop, ks_frac, ks_scale, l1_scale, beta, train_size, valid_size):
        super(SmoothL1_KSLoss, self).__init__()
        self.first_write = True
        self.first_iter = True
        self.ks_stop = ks_stop
        self.ks_frac = ks_frac
        self.ks_scale = ks_scale        
        self.l1_scale = l1_scale
        self.beta = beta
        self.train_size = train_size
        self.valid_size = valid_size
        self.stop = False   
        
    def write_log(self, losses):
        if self.first_write:
            if os.path.exists(self.log_fname): os.remove(self.log_fname)
        
        with open(self.log_fname,'a') as f:
            if self.first_write:
                f.write(f"mean(L1_loss) \t mean(KS_loss) "\
                        f"\t mean(L1_loss)*{self.l1_scale:.3e} \t mean(KS_loss)*{self.ks_scale:.3e}")
            f.write("\n")
            np.savetxt(f, losses.detach().cpu().numpy(), fmt='%.3e', newline="\t")
            
    def forward(self, predictions, targets, write, write_idx):      
        if predictions.is_cuda:
            self.device = torch.device('cuda:%d'%predictions.get_device()) 
        else: self.device = torch.device('cpu')          
        
        #-----SmoothL1----
        l1_loss = 0

        self.beta = 0.1*targets.max()
        diff = predictions-targets
        mask = (diff.abs() < self.beta)
        l1_loss += mask * (0.5*diff**2 / self.beta)
        l1_loss += (~mask) * (diff.abs() - 0.5*self.beta)
           
        #--------KS-------
        distr = torch.distributions.normal.Normal(loc=targets.mean(), 
                                                  scale=targets.std(), 
                                                  validate_args=False)  
        cdf,idx = distr.cdf(targets).flatten().to(self.device).sort()
        
        distr = torch.distributions.normal.Normal(loc=predictions.mean(), 
                                                  scale=predictions.std(), 
                                                  validate_args=False) 
        cdf_pred,idx_pred = distr.cdf(predictions).flatten().to(self.device).sort()     
                                
        vals,idx = targets.flatten().to(self.device).sort()
        vals_pred,idx_pred = predictions.flatten().to(self.device).sort()        

        cdf_interp = Interp1d()(vals_pred, cdf_pred, vals)[0]       
        
        ks_loss = torch.max(torch.abs(cdf-cdf_interp)).to(self.device)           
        
        #--Print Metrics-- 
        if self.first_iter:
            self.maxl1 = l1_loss.mean().detach()                      
            self.maxks = ks_loss.mean().detach()
            self.max_ratio = self.maxks/self.maxl1
                        
            print('---Initial (max) L1, KS losses and their ratio--')
            print(self.maxl1, self.maxks, self.max_ratio)
            print('------------------------------------------------')
            
            self.first_iter = False

        #----Calc Loss----
        #fractions that KS and L1 contribute to the total loss        
        l1_frac = 1-self.ks_frac
        
        loss = (l1_frac*l1_loss.mean()*self.l1_scale +
                self.ks_frac*ks_loss.mean()*self.ks_scale)
        
        #---Save Batch----
        if write_idx == 0:
            if write == 'train': 
                self.batch_size = self.train_size
                self.log_fname = "train_l1ks_log.txt"
            elif write == 'valid': 
                self.batch_size = self.valid_size 
                self.log_fname = "valid_l1ks_log.txt"                
            self.iter_losses_l1 = torch.zeros(self.batch_size,requires_grad=False).to(self.device)
            self.iter_losses_ks = torch.zeros(self.batch_size,requires_grad=False).to(self.device)
        
        self.iter_losses_l1[write_idx] = l1_loss.mean().detach()
        self.iter_losses_ks[write_idx] = ks_loss.mean().detach()
        
        #---Write Logs----
        if write_idx == (self.batch_size-1):
            mean_iter_l1 = self.iter_losses_l1.mean()
            mean_iter_ks = self.iter_losses_ks.mean()
            
            if mean_iter_ks < self.ks_stop or self.stop==True:
                self.stop = True
            
            self.write_log(torch.tensor([mean_iter_l1, mean_iter_ks,
                                         mean_iter_l1*self.l1_scale,
                                         mean_iter_ks*self.ks_scale]))      
            if write == 'valid': self.first_write = False                  
        
        return loss, self.stop
    
class PIMLTurbConfig(EstimatorConfig):
    def __init__(self,
                 n_epochs: int = 1,
                 patience: int = 10,
                 min_delta: float = 1e-5, 
                 logdir: str = "./logs/",
                 lr: float = 1e-3,
                 min_lr = None,                 
                 *args, **kwargs):
        self.n_epochs = n_epochs
        self.logdir = logdir
        self.patience = patience
        self.min_delta = min_delta
        self.lr = lr
        if min_lr==None: self.min_lr = lr*1e-2
        else: self.min_lr = min_lr        
        self.kwargs = kwargs
        self.parameters = {f'model - {k}': v for k, v in self.__dict__.items() if k != 'kwargs'}
        if bool(self.kwargs): self.parameters.update({f'model - {k}': v for k, v in self.kwargs.items()})          
    
class PIMLTurb(TorchBackend):
    def __init__(self, loaders,
                       activ: str = 'Tanhshrink',
                       loss: str = 'SmoothL1_KSLoss',                       
                       ks_stop: float = 0.1,
                       ks_frac: float = 0.5,
                       ks_scale: float = 1,
                       l1_scale: float = 1,                 
                       l1_beta: float = 1,
                       sigma: float = 1,
                       config = PIMLTurbConfig(), 
                       model = PIMLTurbModel()):
        super().__init__(config, model)
        self.config = config
        self.loaders = loaders
        self.device = self.config.kwargs['device']
        self.ks_stop = ks_stop
        self.ks_frac = ks_frac
        self.ks_scale = ks_scale
        self.l1_scale = l1_scale        
        self.beta = l1_beta
        
        x_shape, y_shape = get_loader_shape(self.loaders)                

        self.model = PIMLTurbModel(x_shape[1], y_shape[2]**3, activ, sigma)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)        
        
        if loss == "SmoothL1_KSLoss": 
            self.loss_func = SmoothL1_KSLoss(ks_stop = self.ks_stop, ks_frac = self.ks_frac,
                                             ks_scale = self.ks_scale, l1_scale = self.l1_scale, 
                                             beta = self.beta,
                                             train_size = len(self.loaders['train']), 
                                             valid_size = len(self.loaders['valid']))
        else: 
            print("This network only supports the custom SmoothL1_KSLoss. Please set the 'loss' parameter in config")
            sys.exit()
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                               patience=self.config.patience,
                                                               min_lr=self.config.min_lr) 
        self.config.parameters["model - loss"] = str(self.loss_func).partition("(")[0]
        self.config.parameters["model - activ"] = activ
        
    def write_loss(self, losses, epoch, fname_train = "train_log.txt", fname_valid='valid_log.txt'):
        for idx, fname in enumerate([fname_train, fname_valid]):
            if epoch==1:
                if os.path.exists(fname): os.remove(fname)

            with open(fname,'a') as f:
                if epoch == 1: f.write("epoch \t loss")
                f.write("\n")
                np.savetxt(f, [[losses[0],losses[idx+1]]], fmt='%d \t %.3e', newline="\t")
        
    def train(self):        
        self.model.to(self.device)
        
        if 'cuda' in str(self.device):
            self.optimizer_to(self.optimizer, self.device)      
            
        for epoch in np.linspace(1,int(self.config.n_epochs),int(self.config.n_epochs), dtype='int'):     
            iter_loss = np.zeros(len(self.loaders['train']))
            for idx, (x, y) in enumerate(self.loaders['train']):
                x = x.to(self.device)
                y = y.to(self.device)
                
                y_pred = self.model(x)
                loss, stop = self.loss_func(y_pred, y, 'train', idx)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                iter_loss[idx] = loss.item()
                        
            epoch_loss = iter_loss.mean()
            print(f'train ({epoch}/{int(self.config.n_epochs)}) loss: {epoch_loss:.4e}')
                        
            with torch.set_grad_enabled(False):
                iter_loss_valid = np.zeros(len(self.loaders['valid']))
                for idx, (x, y) in enumerate(self.loaders['valid']):
                    x = x.to(self.device)
                    y = y.to(self.device)

                    y_pred = self.model(x)                                    
                    
                    loss_valid, stop = self.loss_func(y_pred, y, 'valid', idx)

                    iter_loss_valid[idx] = loss_valid.item()

                epoch_loss_valid = iter_loss_valid.mean()
                print(f'valid ({epoch}/{int(self.config.n_epochs)}) loss: {epoch_loss_valid:.4e}')
            
            self.write_loss([epoch, epoch_loss, epoch_loss_valid], epoch)            
            
            if stop:
                print('Reached sufficient KS Loss, stopping...')
                break
                    
            print('-----')     
            
        with open('model_details.txt', 'w') as file:                    
            file.write(f'{str(self.model)}\n\n{str(self.optimizer)}\n\n{str(self.scheduler)}')            
                
        return self.model