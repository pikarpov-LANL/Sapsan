import torch
#import torch.nn.functional as TFunc
import torch.utils.data as Tdata
from torch.autograd import Variable
import sys

class SpacialConvolutionsModel(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(SpacialConvolutionsModel, self).__init__()
    
        self.conv3d = torch.nn.Conv3d(D_in, 72, 2, stride=2, padding=1)
        self.pooling = torch.nn.MaxPool3d(kernel_size=2, padding=1)
        self.conv3d2 = torch.nn.Conv3d(72, 72, 2, stride=2, padding=1)
        self.pooling2 = torch.nn.MaxPool3d(kernel_size=2, padding=1)
        self.conv3d3 = torch.nn.Conv3d(72, 144, 2, stride=2, padding=1)
        self.pooling3 = torch.nn.MaxPool3d(kernel_size=2)

        self.relu = torch.nn.ReLU()
        
        self.linear = torch.nn.Linear(144, 288)
        self.linear2 = torch.nn.Linear(288, D_out)

    
    def forward(self, x):
        c1 = self.conv3d(x)
        p1 = self.pooling(c1)
        c2 = self.conv3d2(self.relu(p1))
        p2 = self.pooling2(c2)

        c3 = self.conv3d3(self.relu(p2))
        p3 = self.pooling3(c3)
        v1 = p3.view(p3.size(0), -1)
        
        '''
        print('original', x.shape)
        print('conv3d', c1.shape)
        print('pooling', p1.shape)
        print('conv3d2', c2.shape)
        print('pooling2', p2.shape)
        print('conv3d3', c3.shape)
        print('pooling3', p3.shape)
        print('pooled size', p3.size(), p3.size(0))
        print('reshape based on pooling', v1.shape) 
        print('cv', ct.shape)
        '''
        
        l1 = self.relu(self.linear(v1))
        l2 = self.linear2(l1)

        #print('ReLU linear', l1.shape)
        #print('linear2', l2.shape)
        #sys.exit()
        
        return l2
