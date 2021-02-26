'''
A set of methods to perform various physical calculations,
such as the Power Spectrum and various analytic turbulence
subgrid models.

-pikarpov
'''

from scipy import fftpack
import numpy as np
from sapsan.utils.plot import line_plot

class PowerSpectrum():
    def __init__(self, u: np.ndarray):
        self.u = u
        self.axis = len(self.u.shape)-1
        self.dim = self.u.shape[1:]
        self.k_bins = None
        self.Ek_bins = None
    
    def kolmogorov(self, kl_A,k):
        return kl_A*k**(-5/3)

    def generate_k(self):
        half = [int(i/2) for i in self.dim]
        k_ar = np.zeros(self.u.shape)

        for a in range(0, half[0]):
            for b in range(0, half[1]):
                for c in range(0, half[2]):
                    grid_points = np.array([[a,b,c],[a,b,-c-1],
                                            [a,-b-1,c],[-a-1,b,c],
                                            [-a-1,-b-1,c],[-a-1,-b-1,-c-1],
                                            [-a-1,b,-c-1],[a,-b-1,-c-1]])

                    for gp in grid_points:
                        k_ar[:,gp[0],gp[1],gp[2]] = [a,b,c]

        k2 = k_ar[0]**2+k_ar[1]**2+k_ar[2]**2
        k = np.sqrt(k2)
        return k
    
    def plot_spectrum(self, kolmogorov=True, kl_A = None):

        if kl_A == None: kl_A = np.amax(self.Ek_bins)*1e1
            
        plt = line_plot([[self.k_bins, self.Ek_bins], [self.k_bins, self.kolmogorov(kl_A, self.k_bins)]], 
                        names = ['data', 'kolmogorov'], plot_type = 'loglog')
        plt.xlim((1e0))
        plt.xlabel('$\mathrm{log(k)}$')
        plt.ylabel('$\mathrm{log(E(k))}$')    
        plt.title('Power Spectrum')
        
        return plt
        
    def calculate(self):
        vk = np.empty((self.u.shape))

        for i in range(self.axis):
            vk[i] = fftpack.fftn(self.u[i]).real
            if i==0: vk2 = vk[0]**2
            else: vk2 += vk[i]**2

        ek = vk2

        k = self.generate_k()

        sort_index = np.argsort(k, axis=None)
        k = np.sort(k, axis=None)
        ek = np.take_along_axis(ek, sort_index, axis=None)

        start = 0
        kmax = int(np.ceil(np.amax(k)))
        self.Ek_bins = np.zeros([kmax+1])

        for i in range(kmax+1):
            for j in range(start, len(ek)):
                if k[j]>i-0.5 and k[j]<=i+0.5:
                    self.Ek_bins[i]+=ek[j]
                    start+=1
                else: break

        self.k_bins = np.arange(kmax+1)
        
        print('Power Spectrum has been calculated. You can access the data through', 
              'PowerSpectrum.k_bins and PowerSpectrum.Ek_bins variables.')
                
class GradientModel():
    def __init__(self, u: np.ndarray, filter_width, delta_u = 1):
        self.u = u
        self.delta_u = delta_u
        self.filter_width = filter_width
        
    def gradient(self):
        gradient_u = np.concatenate((np.gradient(self.u[0,:,:,:], self.delta_u),
                                     np.gradient(self.u[1,:,:,:], self.delta_u),
                                     np.gradient(self.u[2,:,:,:], self.delta_u)),axis=0)
        return gradient_u

    def model(self):
        gradient_u = self.gradient()
        
        dux = gradient_u[0:3]
        duy = gradient_u[3:6]
        return np.array(1/12*self.filter_width**2*dux[2]*duy[2])
        
'''        
class DSModel():
    def __init__(self, dim, fm = 15):
        self.dim = dim
        self.fm = fm
        self.filt = getattr(Filters, 'spectral')
            
    def model(self, vals):

        u = vals[:,:3]
        du = vals[:,3:]
        u = np.moveaxis(u.reshape(self.dim,self.dim,self.dim, u.shape[-1]),-1,0)
        du = np.moveaxis(du.reshape(self.dim,self.dim,self.dim, du.shape[-1]),-1,0)

        L = self.Lvar(u)
        S = self.Stn(du)
        M, Sd = self.Mvar(S)

        Cd = np.zeros((self.dim, self.dim, self.dim))

        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):

                    Cd[i,j,k] = 1/2*((sum((np.matmul(L[:,:,i,j,k],M[:,:,i,j,k])).flatten())/9)/
                                    (sum(np.matmul(M[:,:,i,j,k],M[:,:,i,j,k]).flatten())/9))
        
        tn = np.zeros((3,3,self.dim, self.dim, self.dim))
        for i in range(3):
            for j in range(3):
                tn[i,j]=-2*Cd*Sd*S[i,j]
        return tn
            
    #Dynamic Smagorinsky
    def Lvar(self, u):
        #calculates stress tensor components
        length = len(u)
        tn = np.zeros((length,length,self.dim, self.dim, self.dim))
        for i in range(3):
            for j in range(3):
                tn[i,j] = (self.filt(u[i]*u[j], self.dim)-
                           self.filt(u[i], self.dim)*self.filt(u[j], self.dim))
        return tn

    def Stn(self, du):
        length = 3 #len(u)
        
        S = np.zeros((length,length,self.dim, self.dim, self.dim))
        for i in range(3):
            for j in range(3):
                S[i,j] = 1/2*(du[i]+du[j])
        return S

    def Mvar(self, S):
        length = len(S)
        M = np.zeros((length,length,self.dim, self.dim,self.dim))

        Sd = np.zeros((self.dim, self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    Sd[i,j,k] = np.sqrt(2)*np.linalg.norm(S[:,:,i,j,k])

        for i in range(3):
            for j in range(3):
                #>>>>>>>>1/2 is the ratio of filters!<<<<<<<<<
                M[i,j] = (self.filt(Sd*S[i,j], self.dim) - 
                           (1/2)**2*self.filt(Sd, self.dim)*self.filt(S[i,j], self.dim))
        return M, Sd    
'''