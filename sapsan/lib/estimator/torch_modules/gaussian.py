"""
Gaussian kernel done via 1D convolutions which is done
via PyTorch for a 3D filtering operation within the ML model
"""

import torch 

class Gaussian(torch.nn.Module):
    def __init__(self, sigma=2):
        super(Gaussian, self).__init__()
        self.sigma = sigma
        self.device = torch.device('cpu')

    def make_gaussian_kernel(self):
        ks = int(self.sigma * 5)
        if ks % 2 == 0:
            ks += 1
        ts = torch.linspace(-ks // 2, ks // 2 + 1, ks)
        gauss = torch.exp((-(ts / self.sigma)**2 / 2))
        kernel = gauss / gauss.sum()

        return kernel        
        
    def forward(self, tensor):  
        if tensor.is_cuda:
            self.device = torch.device('cuda:%d'%tensor.get_device()) 
        else: self.device = torch.device('cpu') 
        
        k = self.make_gaussian_kernel()
        
        # Separable 1D convolution
        vol_in = tensor[:]
        k1d = k[None, None, :, None, None].to(self.device)
        for i in range(3):
            vol_in = vol_in.permute(0, 1, 4, 2, 3)                        
            vol_in = torch.nn.functional.conv3d(vol_in, k1d, stride=1, padding=(len(k) // 2, 0, 0))
        vol_3d_sep = vol_in
        
        return vol_3d_sep  