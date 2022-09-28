'''
A set of methods to perform various physical calculations,
such as the Power Spectrum and various analytic turbulence
subgrid models. Currently contains:

functions: tensor
classes: PowerSpectrum  
         GradientModel  
         DynamicSmagorinskyModel 
         picae_func
         
-pikarpov
'''

from scipy import fftpack
import numpy as np
import torch
from sapsan.utils.plot import line_plot
from sapsan.utils.filters import gaussian, spectral

def ReynoldsStress(u, filt=gaussian, filt_size=2, only_x_components = False):
    #calculates stress tensor components

    assert len(u.shape) == 4, "Input variable has to be in the following format: [axis, D, H, W]"
    
    if only_x_components: i_dim = 1
    else: i_dim = 3
    
    rs = np.zeros((i_dim,3,np.shape(u[0])[-3], np.shape(u[0])[-2], np.shape(u[0])[-1]))

    for i in range(i_dim):
        for j in range(3):
            rs[i,j] = filt(u[i]*u[j], filt_size)-filt(u[i], filt_size)*filt(u[j], filt_size)
    if only_x_components: return rs[0]
    else: return rs


class PowerSpectrum():
    def __init__(self, u: np.ndarray):
        self.axis = u.shape[0]
        assert self.axis in [1,2,3], "Input variable has to be in the following format: for 3D [axis, D, H, W], for 2D [axis, D, H], and for 1D [axis, D]"
        self.u = u
        self.dim = self.u.shape[1:]
        self.k_bins = None
        self.Ek_bins = None
    
    def kolmogorov(self, kl_A,k):
        return kl_A*k**(-5/3)

    def generate_k(self):
        half = [int(i/2) for i in self.dim]
        k_ar = np.zeros(self.u.shape, dtype=int)

        if self.axis == 1:
            for a in range(0, half[0]):                
                grid_points = np.array([[a],[-a-1]])
                for gp in grid_points:
                    k_ar[:,gp[0]] = [a]                

        elif self.axis == 2:
            for a in range(0, half[0]):                    
                for b in range(0, half[1]):                
                    grid_points = np.array([[a,b],[a,-b-1],
                                            [-a-1,b],[-a-1,-b-1]])
                    for gp in grid_points:
                        k_ar[:,gp[0],gp[1]] = [a,b]                

        elif self.axis == 3:
            for a in range(0, half[0]):                    
                for b in range(0, half[1]):                                
                    for c in range(0, half[2]):
                        grid_points = np.array([[a,b,c],[a,-b-1,c],
                                                [-a-1,b,c],[-a-1,-b-1,c],
                                                [a,b,-c-1],[-a-1,-b-1,-c-1],
                                                [-a-1,b,-c-1],[a,-b-1,-c-1]])
                        for gp in grid_points:
                            k_ar[:,gp[0],gp[1],gp[2]] = [a,b,c]
                                        
        k2 = np.zeros(self.u.shape[1:])
        for i in range(self.axis):
            k2 += k_ar[i]**2
                    
        k = np.sqrt(k2)

        return k
    
    def spectrum_plot(self, k_bins, Ek_bins, kolmogorov=True, kl_A = None):
        assert len(k_bins.shape) == 1, "k_bins has to be flattened to 1D"
        assert len(Ek_bins.shape) == 1, "Ek_bins has to be flattened to 1D"
        
        if kl_A == None: kl_A = np.amax(Ek_bins)*1e1
            
        ax = line_plot([[k_bins, Ek_bins]], 
                        label = ['data'], 
                        plot_type = 'loglog')
        
        if kolmogorov:
            # does not include the 0th bin (modes below 0.5)
            ax = line_plot([[k_bins[1:], self.kolmogorov(kl_A, k_bins[1:])]], 
                            label = ['kolmogorov'], 
                            plot_type = 'loglog', ax = ax)
        
        ax.set_xlabel('$\mathrm{log(k)}$')
        ax.set_ylabel('$\mathrm{log(E(k))}$')    
        ax.set_title('Power Spectrum')
        
        return ax
        
    def calculate(self):
        uk = np.zeros((self.u.shape))
        Ek = np.zeros((self.u.shape[1:]))

        # mode (k) grid
        k = self.generate_k()

        # fourier transform of u to get kinetic energy E(k)   
        for i in range(self.axis):
            uk[i] = fftpack.fftn(self.u[i]).real
            Ek += uk[i]**2      

        sort_index = np.argsort(k, axis=None)
        k = np.sort(k, axis=None)
        Ek = np.take_along_axis(Ek, sort_index, axis=None)

        start = 0
        kmax = int(np.ceil(np.amax(k)))
        Ek_bins = np.zeros([kmax+1])

        for i in range(kmax+1):
            for j in range(start, len(Ek)):
                if k[j]>i-0.5 and k[j]<=i+0.5:
                    Ek_bins[i] += Ek[j]**2
                    start+=1
                else: break
            Ek_bins[i] = np.sqrt(Ek_bins[i])

        k_bins = np.arange(kmax+1)
        
        print('Power Spectrum has been calculated. k and E(k) have been returned.')
        
        return k_bins, Ek_bins
        

class GradientModel():
    def __init__(self, u: np.ndarray, filter_width, delta_u = 1):
        assert len(u.shape) == 4, "Input variable has to be in the following format: [axis, D, H, W]"

        self.u = u
        self.delta_u = delta_u
        self.filter_width = filter_width

        print("Calculating the tensor from Gradient model...")
        print("Note: input variables have to be filtered!")

    def gradient(self):

        gradient_u = np.stack((np.gradient(self.u[0,:,:,:], self.delta_u),
                               np.gradient(self.u[1,:,:,:], self.delta_u),
                               np.gradient(self.u[2,:,:,:], self.delta_u)),axis=0)
        return gradient_u

    def model(self):
        gradient_u = self.gradient()

        tn = np.zeros(gradient_u.shape)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    tn[i,j] += np.multiply(gradient_u[i,k], gradient_u[j,k])

        tn = 1/12*self.filter_width**2*tn

        print('Tensor by the gradient model has the shape: [column, row, D, H, W]')
        print('As calculated: ', tn.shape)

        return tn

    
class DynamicSmagorinskyModel():    
    def __init__(self, u, filt=spectral, original_filt_size = 15, filt_ratio = 0.5, **kwargs):     

        self.axis = u.shape[0]
        assert self.axis in [2,3], "Input variable has to be in the following format: for 3D [axis, D, H, W], and for 2D [axis, D, H]"
        
        self.u = u
        self.filt = filt
        self.filt_ratio = filt_ratio
        self.filt_size = int(np.floor(original_filt_size*self.filt_ratio))
        self.kwargs = kwargs
        
        print("Calculating the tensor from Dynamic Smagorinsky model...")
        print("Note: input variables have to be filtered!")

    def model(self):
        if "du" in self.kwargs: du = self.kwargs["du"]
        else:
            print('Derivative was not provided: will be calculated via np.gradient()')
            if "delta_u" in self.kwargs: delta_u = self.kwargs['delta_u']
            else:
                print("delta_u (spacing between values) was not provided: setting delta_u = 1")
                delta_u = 1
            
            if self.axis==2:
                du = np.stack((np.gradient(self.u[0,:,:], delta_u),
                               np.gradient(self.u[1,:,:], delta_u)),axis=0) 
            else:
                du = np.stack((np.gradient(self.u[0,:,:,:], delta_u),
                               np.gradient(self.u[1,:,:,:], delta_u),
                               np.gradient(self.u[2,:,:,:], delta_u)),axis=0) 
        self.shape = du.shape

        L = self.Lvar(self.u)
        S = self.Stn(du)
        M, Sd = self.Mvar(S)

        Cd = np.zeros(self.shape[-self.axis:])

        for i in range(self.shape[-self.axis]):
            for j in range(self.shape[-(self.axis-1)]):                
                if self.axis == 2:
                    Cd[i,j] = 1/2*((sum((np.matmul(L[:,:,i,j],M[:,:,i,j])).flatten())/4)/
                                   (sum(np.matmul(M[:,:,i,j],M[:,:,i,j]).flatten())/4))
                    continue
                for k in range(self.shape[-1]):
                    Cd[i,j,k] = 1/2*((sum((np.matmul(L[:,:,i,j,k],M[:,:,i,j,k])).flatten())/9)/
                                     (sum(np.matmul(M[:,:,i,j,k],M[:,:,i,j,k]).flatten())/9))

        tn = np.zeros(self.shape)
        for i in range(self.axis):
            for j in range(self.axis):
                tn[i,j]=-2*Cd*Sd*S[i,j]
        return tn

    def Lvar(self, u):
        #calculates stress tensor components
        tn = np.zeros(self.shape)
        for i in range(self.axis):
            for j in range(self.axis):
                tn[i,j] = (self.filt(u[i]*u[j], self.filt_size)-
                           self.filt(u[i], self.filt_size)*self.filt(u[j], self.filt_size))
        return tn

    def Stn(self, du):
        S = np.zeros(self.shape)

        for i in range(self.axis):
            for j in range(self.axis):
                S[i,j] = 1/2*(du[i,j]+du[j,i])
        return S

    def Mvar(self, S):
        length = len(S)
        M = np.zeros(self.shape)

        Sd = np.zeros(self.shape[-self.axis:])
        for i in range(self.shape[-self.axis]):
            for j in range(self.shape[-(self.axis-1)]):
                if self.axis==2:
                    Sd[i,j] = np.sqrt(2)*np.linalg.norm(S[:,:,i,j])
                    continue
                for k in range(self.shape[-1]):
                    Sd[i,j,k] = np.sqrt(2)*np.linalg.norm(S[:,:,i,j,k])

        for i in range(self.axis):
            for j in range(self.axis):
                M[i,j] = (self.filt(Sd*S[i,j], self.filt_size) -
                         (self.filt_ratio)**2*self.filt(Sd, self.filt_size)*self.filt(S[i,j], self.filt_size))
        return M, Sd 
    
    
class picae_func():
    def minmaxscaler(data):
        """ scale large turbulence dataset by channel"""
        nsnaps = data.shape[0]
        dim = data.shape[1]
        nch = data.shape[4]

        #scale per channel
        data_scaled = []
        rescale_coeffs = []
        for i in range(nch):
            data_ch = data[:,:,:,:,i]
            minval = data_ch.min(axis=0)
            maxval = data_ch.max(axis=0)
            temp = (data_ch - minval)/(maxval - minval)
            data_scaled.append(temp)
            rescale_coeffs.append((minval,maxval))
        data_scaled = np.stack(data_scaled, axis=4)
        np.save('rescale_coeffs_3DHIT', rescale_coeffs)
        return data_scaled

    def inverse_minmaxscaler(data,filename):
        """ Invert scaling using previously saved minmax coefficients """
        rescale_coeffs = np.load(filename)
        nsnaps = data.shape[0]
        dim = data.shape[1]
        nch = data.shape[4]

        #scale per channel
        data_orig = []
        for i in range(nch):
            data_ch = data[:,:,:,:,i]
            (minval, maxval) = rescale_coeffs[i]
            temp = data_ch*(maxval - minval) + minval
            data_orig.append(temp)
        data_orig = np.stack(data_orig, axis=4)
        return data_orig

    def convert_to_torchchannel(data):
        """ converts from  [snaps,dim1,dim2,dim3,nch] ndarray to [snaps,nch,dim1,dim2,dim3] torch tensor"""
        nsnaps = data.shape[0]
        dim1, dim2, dim3 = data.shape[1], data.shape[2], data.shape[3] 
        nch = data.shape[-1] #nch is last dimension in numpy input
        torch_permuted = np.zeros((nsnaps, nch, dim1, dim2, dim3))
        for i in range(nch):
            torch_permuted[:,i,:,:,:] = data[:,:,:,:,i]
        torch_permuted = torch.from_numpy(torch_permuted)
        return torch_permuted


    def convert_to_numpychannel_fromtorch(tensor):
        """ converts from [snaps,nch,dim1,dim2,dim3] torch tensor to [snaps,dim1,dim2,dim3,nch] ndarray """
        nsnaps = tensor.size(0)
        dim1, dim2, dim3 = tensor.size(2), tensor.size(3), tensor.size(4)
        nch = tensor.size(1)
        numpy_permuted = torch.zeros(nsnaps, dim1, dim2, dim3, nch)
        for i in range(nch):
            numpy_permuted[:,:,:,:,i] = tensor[:,i,:,:,:]
        numpy_permuted = numpy_permuted.numpy()
        return numpy_permuted    

    def np_divergence(flow,grid):
        np_Udiv = np.gradient(flow[:,:,:,0], grid[0])[0]
        np_Vdiv = np.gradient(flow[:,:,:,1], grid[1])[1]
        np_Wdiv = np.gradient(flow[:,:,:,2], grid[2])[2]
        np_div = np_Udiv + np_Vdiv + np_Wdiv
        total = np.sum(np_div)/(np.power(128,3))
        return total

    def calcDivergence(flow,grid):
        flow = convert_to_numpychannel_fromtorch(flow.detach())
        field = flow[0,::]
        np_Udiv = np.gradient(field[:,:,:,0], grid[0])[0]
        np_Vdiv = np.gradient(field[:,:,:,1], grid[1])[1]
        np_Wdiv = np.gradient(field[:,:,:,2], grid[2])[2]
        np_div = np_Udiv + np_Vdiv + np_Wdiv
        total = np.abs(np.sum(np_div)/(np.power(128,3)))
        return total

    def divergence_diff(dns,model,grid):
        """ Computes difference between DNS divergence and model divergence"""
        #DNS
        dns  = dns[::].clone()
        sampleDNS = dns.cpu()
        np_sampleDNS = convert_to_numpychannel_fromtorch(sampleDNS)
        DNSDiv =  np_divergence(np_sampleDNS[0,::],grid)
        #Model
        model  = model[::].clone()
        sampleMOD = model.detach().cpu()
        np_sampleMOD = convert_to_numpychannel_fromtorch(sampleMOD)
        MODDiv =  np_divergence(np_sampleMOD[0,::],grid)
        #difference
        diff = np.abs(np.abs(DNSDiv) - np.abs(MODDiv))
        return diff    