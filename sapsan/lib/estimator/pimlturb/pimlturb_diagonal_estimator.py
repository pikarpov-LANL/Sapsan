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
    
    Corresponding 'losses_train_log.txt' and 'losses_valid_log.txt'
    are written out to include the individual loss evolutions.
    '''    
    def __init__(self, ks_stop, scalel1, beta):
        super(SmoothL1_KSLoss, self).__init__()
        self.counter = 1
        self.first_write = 9
        self.ks_stop = ks_stop
        self.scalel1 = scalel1
        self.beta = beta
        self.stop = False        
        
    def write_loss(self, losses, fname = "losses_train_log.txt"):
        if self.counter==self.first_write:
            if os.path.exists(fname): os.remove(fname)
        
        with open(fname,'a') as f:
            if self.counter<=self.first_write: 
                f.write("mean(L1_loss) \t mean(KS_loss) \t norm(L1_loss) \t norm(KS_loss) \t norm(L1_loss)*%.3e"%self.scaling)
            f.write("\n")
            np.savetxt(f, losses.detach().cpu().numpy(), fmt='%.3e', newline="\t")
            
    def write_valid(self, losses, fname = "losses_valid_log.txt"):
        if self.counter==self.first_write+1:
            if os.path.exists(fname): os.remove(fname)
        
        with open(fname,'a') as f:
            if self.counter<=self.first_write+1: 
                f.write("mean(L1_valid) \t mean(KS_valid)")
            f.write("\n")
            np.savetxt(f, losses.detach().cpu().numpy(), fmt='%.3e', newline="\t")
        
    def forward(self, predictions, targets):                   
        try: 
            self.device = predictions.get_device()
            if self.device==-1: self.device=torch.device('cpu')
        except: self.device=torch.device('cpu')

        size = list(targets.size())[0]  
        lossks= torch.zeros(size).to(self.device)
        
        #-----SmoothL1----
        lossl1 = 0

        self.beta = 0.1*targets.max()
        diff = predictions-targets
        mask = (diff.abs() < self.beta)
        lossl1 += mask * (0.5*diff**2 / self.beta)
        lossl1 += (~mask) * (diff.abs() - 0.5*self.beta)
        
        for i in range(size):                        
            #-------KS------
            distr = torch.distributions.normal.Normal(loc=0, scale=1, validate_args=False)        

            cdf = distr.cdf(targets[i]).to(self.device)
            cdf_pred = distr.cdf(predictions[i]).to(self.device)

            cdf, idx = cdf.flatten().to(self.device).sort()
            cdf_pred, idx_pred = cdf_pred.flatten().to(self.device).sort()

            y = torch.linspace(0,1,list(cdf.size())[0]).to(self.device)

            y_new = Interp1d()(cdf, y, cdf_pred)[0]
            
            ks = torch.max(torch.abs(y-y_new)).to(self.device)         
            
            lossks[i] = ks
        
        if self.counter == 1:   
            self.iter_losses_l1 = torch.zeros(self.first_write,requires_grad=False).to(self.device)
            self.iter_losses_ks = torch.zeros(self.first_write,requires_grad=False).to(self.device)
            
            self.maxl1 = lossl1.mean().detach()                      
            self.maxks = lossks.mean().detach()
            self.max_ratio = self.maxks/self.maxl1
                        
            print('---Max L1, KS losses and their ratio--')
            print(self.maxl1, self.maxks, self.max_ratio)
            print('--------------------------------------')
            
            self.scaling = self.maxl1*self.scalel1                                

        fracks = 0.5
        fracl1 = 1 - fracks        
        
        loss = (fracl1*lossl1.mean()/self.maxl1*self.scaling +
                fracks*lossks.mean()/self.maxks)
        #----------------
        if (self.counter % 10) != 0: 
            self.iter_losses_l1[(self.counter % 10)-1] = lossl1.mean().detach()
            self.iter_losses_ks[(self.counter % 10)-1] = lossks.mean().detach()
        #----------------     
        
        if (self.counter % 10) == self.first_write:
            mean_iter_l1 = self.iter_losses_l1.mean()
            mean_iter_ks = self.iter_losses_ks.mean()
            
            if mean_iter_ks < self.ks_stop or self.stop==True:
                self.stop = True
                
            mean_l1_valid = lossl1.mean()
            mean_ks_valid = lossks.mean()
            self.write_loss(torch.tensor([mean_iter_l1, mean_iter_ks,
                                          mean_iter_l1/self.maxl1,
                                          mean_iter_ks/self.maxks,
                                          mean_iter_l1/self.maxl1*self.scaling]))      
            self.iter_losses_l1.fill_(0)
            self.iter_losses_ks.fill_(0) 
                                                
        if (self.counter % 10) == 0:
            mean_l1_valid = lossl1.mean()
            mean_ks_valid = lossks.mean()
            self.write_valid(torch.tensor([mean_l1_valid, mean_ks_valid]))                  
        
        self.counter +=1           
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
    def __init__(self, activ, loss,
                       loaders,
                       ks_stop = 0.1,
                       scalel1 = 1e6,
                       betal1 = 1,
                       sigma = 1,
                       config = PIMLTurbConfig(), 
                       model = PIMLTurbModel()):
        super().__init__(config, model)
        self.config = config
        self.loaders = loaders
        self.device = self.config.kwargs['device']
        self.ks_stop = ks_stop
        self.scalel1 = scalel1
        self.beta = betal1
        
        x_shape, y_shape = get_loader_shape(self.loaders)
        
        self.model = PIMLTurbModel(x_shape[1], y_shape[2]**3, activ, sigma)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)        
        
        if loss == "SmoothL1_KSLoss": self.loss_func = SmoothL1_KSLoss(self.ks_stop, self.scalel1, self.beta)
        else: 
            print("This network only supports the custom SmoothL1_KSLoss. Please set the 'loss' parameter in config")
            sys.exit()
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                               patience=self.config.patience,
                                                               min_lr=self.config.min_lr) 
        self.config.parameters["model - loss"] = str(self.loss_func).partition("(")[0]
        self.config.parameters["model - activ"] = activ
        
    def write_loss(self, losses, epoch, fname = "train_log.txt", fname_valid='valid_log.txt'):
        if epoch==1:
            if os.path.exists(fname): os.remove(fname)
            if os.path.exists(fname_valid): os.remove(fname_valid)
        
        with open(fname,'a') as f:
            if epoch == 1: f.write("epoch \t train_loss")
            f.write("\n")
            np.savetxt(f, [[losses[0],losses[1]]], fmt='%d \t %.3e', newline="\t")
            
        with open(fname_valid,'a') as f:
            if epoch == 1: f.write("epoch \t valid_loss")
            f.write("\n")
            np.savetxt(f, [[losses[0],losses[2]]], fmt='%d \t %.3e', newline="\t")
        
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
                loss, stop = self.loss_func(y_pred, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                iter_loss[idx] = loss.item()
                        
            epoch_loss = iter_loss.mean()
            print(f'train ({epoch}/{int(self.config.n_epochs)}) loss: {epoch_loss}')                        
                        
            with torch.set_grad_enabled(False):
                iter_loss_valid = np.zeros(len(self.loaders['valid']))
                for idx, (x, y) in enumerate(self.loaders['valid']):
                    x = x.to(self.device)
                    y = y.to(self.device)

                    y_pred = self.model(x)                                    
                    
                    loss_valid, stop = self.loss_func(y_pred, y)
    
                    iter_loss_valid[idx] = loss_valid.item()

                epoch_loss_valid = iter_loss_valid.mean()
                print(f'valid ({epoch}/{int(self.config.n_epochs)}) loss: {epoch_loss_valid}')
            
            self.write_loss([epoch, epoch_loss, epoch_loss_valid], epoch)            
            
            if stop:
                print('Reached sufficient KS Loss, stopping...')
                break
                    
            print('---')            
                
        return self.model