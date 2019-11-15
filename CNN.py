import torch
import torch.nn.functional as TFunc
import torch.utils.data as Tdata
from torch.autograd import Variable

class SpacialConvolutionsModel(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(SpacialConvolutionsModel, self).__init__()
    
        self.conv3d = torch.nn.Conv3d(D_in, 72, 2, stride=2)
        self.pooling = torch.nn.MaxPool3d(kernel_size=2)
        self.conv3d2 = torch.nn.Conv3d(72, 144, 2, stride=2)
        self.pooling2 = torch.nn.MaxPool3d(kernel_size=2)

        self.relu = torch.nn.ReLU()
        
        self.linear = torch.nn.Linear(144, 300)
        self.linear2 = torch.nn.Linear(300, D_out)

    
    def forward(self, x):
        convolved = self.conv3d(x)
        pooled = self.pooling(convolved)
        
        relu = self.relu(pooled)
        
        convolved2 = self.conv3d2(relu)
        pooled2 = self.pooling2(convolved2)
        
        pooled2 = pooled2.view(pooled.size(0), -1)
        
        l1 = TFunc.relu(self.linear(pooled2))
        l2 = self.linear2(l1)
        
        return l2
