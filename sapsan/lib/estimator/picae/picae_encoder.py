"""
Convolutional Auto Encoder with Divergence-Free Kernel and with periodic padding

The model is based on A.T.Mohan, 2020arXiv200200021M
"""

import json
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader

from sapsan.core.models import EstimatorConfig
from sapsan.lib.estimator.pytorch_estimator import TorchEstimator
from sapsan.lib.data import get_loader_shape


class PICAEModel(torch.nn.Module): 
    """
    Init and define stacked Conv Autoencoder layers and Physics layers 
    """
    def __init__(self, input_dim = (128,128,128), 
                       input_size = 3, 
                       batch = 4, 
                       nfilters = 6, 
                       kernel_size = (3,3,3), 
                       enc_nlayers = 3, 
                       dec_nlayers = 3, 
                       device = torch.device('cpu')):        
        super(PICAEModel, self).__init__()
        self.il = input_dim[0]
        self.jl = input_dim[1]
        self.kl = input_dim[2]
        self.input_size= input_size # no. of channels
        self.nfilters = nfilters
        self.batch= batch
        self.kernel_size = kernel_size
        self.output_size = 6 #6 gradient components for 3 vector components of a CURL
        self.encoder_nlayers= enc_nlayers
        self.decoder_nlayers= dec_nlayers
        self.total_layers = self.encoder_nlayers + self.decoder_nlayers
        self.outlayer_padding = kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2
        self.device = device
        
        self.te = TorchEstimator

        ############## Define Encoder layers
        encoder_cell_list=[]
        for layer in range(self.encoder_nlayers):
            if layer == 0:
                cell_inpsize = self.input_size
                cell_outputsize = self.nfilters
                stridelen = 2
            else:
                cell_inpsize = self.nfilters
                cell_outputsize = self.nfilters
                stridelen = 2
       
            encoder_cell_list.append(self.te.to_device(self, torch.nn.Conv3d(
                                                             out_channels = cell_outputsize, 
                                                             in_channels = cell_inpsize, 
                                                             kernel_size = self.kernel_size, 
                                                             stride = stridelen)))

        # accumulate layers
        self.encoder_cell_list = torch.nn.ModuleList(encoder_cell_list)

        ############## Define Decoder layers
        decoder_cell_list=[]
        for layer in range(self.decoder_nlayers):
            if layer == (self.decoder_nlayers -1):
                cell_inpsize = self.nfilters
                cell_outputsize = self.input_size
                cell_padding = 0
                stridelen = 2
            else:
                cell_inpsize = self.nfilters
                cell_outputsize = self.nfilters
                cell_padding = 0
                stridelen = 2        
            
            decoder_cell_list.append(self.te.to_device(self, torch.nn.ConvTranspose3d(
                                                              out_channels = cell_outputsize, 
                                                              in_channels = cell_inpsize,
                                                              kernel_size = self.kernel_size, 
                                                              stride = stridelen, 
                                                              output_padding = cell_padding)))

        # accumulate layers
        self.decoder_cell_list = torch.nn.ModuleList(decoder_cell_list)
        
        ############# Physics Layers
        #d/dx
        self.ddxKernel = torch.zeros(3, 3, 3)
        self.ddxKernel[1,0,1] = -0.5
        self.ddxKernel[1,2,1] = 0.5
        #d/dy
        self.ddyKernel = torch.zeros(3, 3, 3)
        self.ddyKernel[0,1,1] = -0.5
        self.ddyKernel[2,1,1] = 0.5
        #d/dz
        self.ddzKernel = torch.zeros(3, 3, 3)
        self.ddzKernel[1,1,0] = -0.5
        self.ddzKernel[1,1,2] = 0.5
        #### declare weights
        self.weights = torch.zeros((self.output_size, self.input_size, 3, 3, 3))
        self.weights = self.weights.type(self.te.tensor_to_device(self))
        #dfy/dx
        self.weights[0,0,::] = torch.zeros(3, 3, 3)
        self.weights[0,1,::] = self.ddxKernel.clone()
        self.weights[0,2,::] = torch.zeros(3, 3, 3)
        #dfz/dx
        self.weights[1,0,::] = torch.zeros(3, 3, 3)
        self.weights[1,1,::] = torch.zeros(3, 3, 3)
        self.weights[1,2,::] = self.ddxKernel.clone()
        #dfx_dy
        self.weights[2,0,::] = self.ddyKernel.clone()
        self.weights[2,1,::] = torch.zeros(3, 3, 3)
        self.weights[2,2,::] = torch.zeros(3, 3, 3)
        #dfz_dy
        self.weights[3,0,::] = torch.zeros(3, 3, 3)
        self.weights[3,1,::] = torch.zeros(3, 3, 3)
        self.weights[3,2,::] = self.ddyKernel.clone()
        #dfx_dz
        self.weights[4,0,::] = self.ddzKernel.clone()
        self.weights[4,1,::] = torch.zeros(3, 3, 3)
        self.weights[4,2,::] = torch.zeros(3, 3, 3)
        #dfy_dz
        self.weights[5,0,::] = torch.zeros(3, 3, 3)
        self.weights[5,1,::] = self.ddzKernel.clone()
        self.weights[5,2,::] = torch.zeros(3, 3, 3)        
        ### define curl operation
        self.rep_pad = torch.nn.ReplicationPad3d(1)
        self.curlConv = torch.nn.Conv3d(self.input_size,self.output_size,3,bias=False,padding=0)
        with torch.no_grad():
            self.curlConv.weight = torch.nn.Parameter(self.weights)
        self.curlField = torch.zeros([self.batch,self.input_size,self.il,self.jl,self.kl])
        self.curlField = self.curlField.type(self.te.tensor_to_device(self))
        self.register_buffer('r_curlField', self.curlField)
        self.curlGrad = torch.zeros([self.batch,self.output_size,self.il,self.jl,self.kl])
        self.curlGrad = self.curlGrad.type(self.te.tensor_to_device(self))    
        self.register_buffer('r_curlGrad', self.curlGrad)        
        
    def padHITperiodic(self,field):
        oldSize = field.size(-1)
        newSize = oldSize + 3 #2nd order accurate periodic padding at boundary
        newField = torch.zeros(self.batch,self.input_size,newSize,newSize,newSize).type(self.te.tensor_to_device(self))
        
        # fill interior cells
        newField[:,:,:-3,:-3,:-3] = field
        # fill boundary cells with periodic values for 2nd order difference
        #Ghost Cell N+1
        newField[:,:,-3,::] = newField[:,:,0,::] # i axis
        newField[:,:,:,-3,:] = newField[:,:,:,0,:] # j axis
        newField[:,:,:,:,-3] = newField[:,:,:,:,0] # k axis
        #Ghost Cell N+2
        newField[:,:,-2,::] = newField[:,:,1,::] # i axis
        newField[:,:,:,-2,:] = newField[:,:,:,1,:] # j axis
        newField[:,:,:,:,-2] = newField[:,:,:,:,1] # k axis
        #Ghost Cell N+3
        newField[:,:,-1,::] = newField[:,:,2,::] # i axis
        newField[:,:,:,-1,:] = newField[:,:,:,2,:] # j axis
        newField[:,:,:,:,-1] = newField[:,:,:,:,2] # k axis
        return newField      

    def forward(self,x):
        x = x.float()
        
        cur_input = x
        
        # Encoder
        for layer in range(self.encoder_nlayers):
            x = self.encoder_cell_list[layer](x)
            
        # Decoder
        for layer in range(self.decoder_nlayers):
            x = self.decoder_cell_list[layer](x)
            
        # Physics Layers
        x = self.padHITperiodic(x) # PADDING with periodic BC
        curlGrad = self.curlConv(x) # compute conv
        curlField = torch.zeros([self.batch,self.input_size,self.il,self.jl,self.kl])
        curlField = curlField.type(self.te.tensor_to_device(self))
        
        #construct curl vector
        curlField[:,0,::] = curlGrad[:,3,::] - curlGrad[:,5,::]
        curlField[:,1,::] = curlGrad[:,4,::] - curlGrad[:,1,::]
        curlField[:,2,::] = curlGrad[:,0,::] - curlGrad[:,2,::]            

        #---> deleted 'x' output
        return curlField
        
        
class PICAEConfig(EstimatorConfig):
    
    # set defaults per your liking, add more parameters
    def __init__(self, nfilters = 6, 
                       kernel_size = (3,3,3), 
                       enc_nlayers = 3, 
                       dec_nlayers = 3,                    
                       n_epochs = 1,
                       patience: int = 10,
                       min_delta: float = 1e-5, 
                       logdir: str = "./logs/",
                       *args, **kwargs):
        self.nfilters = nfilters
        self.kernel_size = kernel_size
        self.enc_nlayers = enc_nlayers
        self.dec_nlayers = dec_nlayers
        self.n_epochs = n_epochs
        self.logdir = logdir
        self.patience = patience
        self.min_delta = min_delta
        self.kwargs = kwargs
        
        #a few custom names for parameters to record in mlflow
        self.parameters = {
                        "model - n_epochs": self.n_epochs,
                        "model - min_delta": self.min_delta,
                        "model - patience": self.patience,
                    }

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            cfg = json.load(f)
            return cls(**cfg)

    def to_dict(self):
        return self.parameters    
    
    
    
class PICAE(TorchEstimator):
    def __init__(self, config = PICAEConfig(), 
                       model = PICAEModel()):
        super().__init__(config, model)
        self.config = config        
    
    def train(self, loaders):
                            
        train_shape, valid_shape = np.array(get_loader_shape(loaders))     

        model = PICAEModel(input_dim = train_shape[2:], 
                           input_size = train_shape[1], 
                           batch = train_shape[0], 
                           nfilters = self.config.nfilters, 
                           kernel_size = self.config.kernel_size, 
                           enc_nlayers = self.config.enc_nlayers, 
                           dec_nlayers = self.config.dec_nlayers,
                           device = self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, weight_decay=1e-5)
        loss_func = torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               patience=3,
                                                               min_lr=1e-5) 

        trained_model = self.torch_train(loaders, model, 
                                         optimizer, loss_func, scheduler, 
                                         self.config)        
        
        return trained_model