from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import h5py as h5
from scipy import fftpack
import matplotlib.colors as colors
from pylab import MaxNLocator


class functions(object):    
    def __init__(self, path='/raid1/JHTDB', dataset='iso', fm=15, filtname='spectral', 
                 dim=128, max_dim=512, t=0, savepath=None, tnc=2, axis=2, sigma=2):
        self.path = path
        self.dataset = dataset
        self.fm = fm
        self.dim = dim
        self.max_dim = max_dim
        self.t = t
        self.axis = axis
        self.filtname = filtname
        self.filt = getattr(self, self.filtname)
        self.savepath = savepath
        self.tnc = tnc
        self.sigma = sigma
        if self.savepath:
            self.Check_Directories()
            

    def spectral(self, im, dim):
        #Spectral Filter
        form = np.shape(im)
        
        #print('form', form)
        #print(self.axis)
        
        half = np.zeros(len(form))
        
        im_fft = fftpack.fftn(im)
        
        #print(im_fft)
        plt.figure()
        if self.axis==3: plot = plt.imshow(np.transpose(im[0,:,:]),cmap='seismic')
        else: plot = plt.imshow(np.transpose(im),cmap='seismic')
        cbar = plt.colorbar(plot)
        
        #plt.figure()
        #plot = plt.imshow(np.transpose(im_fft.real[int(form[0]/2),:,:]),cmap='seismic')
        #cbar = plt.colorbar(plot)
        
        #im_fft = np.fft.fftshift(im_fft)
        
        plt.figure()
        if self.axis==3: plot = plt.imshow(np.transpose(im_fft.real[int(form[0]/2)-1,:,:]),cmap='seismic')
        else: plot = plt.imshow(np.transpose(im_fft.real)[50:78,50:78],vmax = 5e3, cmap='seismic')

        cbar = plt.colorbar(plot)
        
        half[0] = int(form[0]/2)
        half[1] = int(form[1]/2)
        if self.axis==3: half[2] = int(form[2]/2)
        half = [int(i) for i in half]
        
        '''
        temp = np.zeros(np.shape(im_fft), dtype=complex)
        if self.axis==2: temp[half[0]-self.fm:half[0]+self.fm,
                              half[1]-self.fm:half[1]+self.fm]=im_fft[half[0]-self.fm:half[0]+self.fm,
                                                                      half[1]-self.fm:half[1]+self.fm]
        else:  temp[half[0]-self.fm:half[0]+self.fm,
                    half[1]-self.fm:half[1]+self.fm,
                    half[2]-self.fm:half[2]+self.fm]=im_fft[half[0]-self.fm:half[0]+self.fm,
                                                            half[1]-self.fm:half[1]+self.fm,
                                                            half[2]-self.fm:half[2]+self.fm]
        '''
        temp = np.zeros(np.shape(im_fft), dtype=complex)
        corners = [[0, self.fm, 0, self.fm],
                   [temp.shape[0]-self.fm, temp.shape[0], temp.shape[1]-self.fm, temp.shape[1]],
                   [temp.shape[0]-self.fm, temp.shape[0], 0, self.fm],
                   [0, self.fm, temp.shape[1]-self.fm, temp.shape[1]]]
        for c in corners:
            #print(c)
            temp[c[0]:c[1], c[2]:c[3]] = im_fft[c[0]:c[1], c[2]:c[3]]
            for i in range(self.fm):
                for j in range(self.fm):
                    #if c[1]==self.fm and c[3]==self.fm: xi = i; yj = j
                    #if c[1]==temp.shape[0] and c[3]==temp.shape[0]: xi = temp.shape[0]-i-1; yj = temp.shape[0]-j-1
                    #if c[1]==temp.shape[0] and c[3]==self.fm: xi = temp.shape[0]-i-1; yj = j
                    #if c[1]==self.fm and c[3]==temp.shape[0]: xi = i; yj = temp.shape[0]-j-1             
                    if c[1]==temp.shape[0]: xi = c[1]-c[0]-i-1
                    else: xi = i
                    if c[3]==temp.shape[0]: yj = c[3]-c[2]-j-1
                    else: yj = j
                    
                    if np.sqrt((xi-0.5)**2+(yj-0.5)**2)>self.fm: 
                        #print(xi, yj)
                        temp[c[0]+i,c[2]+j]=0
                    if np.sqrt((xi-0.5)**2+(yj-0.5)**2)>self.fm: temp[c[0]+i,c[2]+j]=0                
        '''
  
            temp[0:self.fm, 0:self.fm]=im_fft[0:self.fm, 0:self.fm]
            temp[-self.fm:, -self.fm:]=im_fft[-self.fm:, -self.fm:]
            temp[-self.fm:, 0:self.fm]=im_fft[-self.fm:, 0:self.fm]
            temp[0:self.fm, -self.fm:]=im_fft[0:self.fm, -self.fm:]
            for i in range(self.fm,self.fm):
                for j in range(self.fm,self.fm):
                    if np.sqrt((i-0.5)**2+(j-0.5)**2)>self.fm: temp[i,j]=0
                    if np.sqrt((i-0.5)**2+(j-0.5)**2)>self.fm: temp[i,j]=0
        '''
        
        '''
        plt.figure()
        if self.axis==3: plot = plt.imshow(np.transpose(temp.real[int(form[0]/2)-1,:,:]),cmap='seismic')
        else: plot = plt.imshow(np.transpose(temp.real)[50:78,50:78],vmax = 5e3, cmap='seismic')
        cbar = plt.colorbar(plot)
        for i in range(half[0]-self.fm,half[0]+self.fm):
            for j in range(half[1]-self.fm,half[1]+self.fm):
                if self.axis==2:
                    if np.sqrt((half[0]-i-0.5)**2+(half[1]-j-0.5)**2)>self.fm: temp[i,j]=0
                else:
                    for k in range(half[2]-self.fm,half[2]+self.fm):
                        if np.sqrt((half[0]-i-0.5)**2+(half[1]-j-0.5)**2+(half[2]-k-0.5)**2)>self.fm: temp[i,j,k]=0
                        '''
        '''
        plt.figure()
        if self.axis==3: plot = plt.imshow(np.transpose(temp.real[int(form[0]/2)-1,:,:]),cmap='seismic')
        #else: plot = plt.imshow(np.transpose(temp.real)[50:78,50:78],vmax = 5e3, cmap='seismic')
        else: plot = plt.imshow(np.transpose(temp.real)[0:15,0:15],vmax = 5e3, cmap='seismic')
        cbar = plt.colorbar(plot)
        
        for c in corners:
            #print(c)
            plt.figure()
            plot = plt.imshow(np.transpose(temp.real)[c[0]:c[1], c[2]:c[3]],vmax = 5e3, cmap='seismic')
            cbar = plt.colorbar(plot)
            
            plt.figure()
            plot = plt.imshow(np.transpose(im_fft.real)[c[0]:c[1], c[2]:c[3]],vmax = 5e3, cmap='seismic')
            cbar = plt.colorbar(plot)
        '''
        
        #im_fft = np.fft.ifftshift(temp)
        #im_fft = np.fft.ifftshift(im_fft)
        
        #plt.figure()
        #if self.axis==3: plot = plt.imshow(np.transpose(im_fft.real[0,:,:]),cmap='seismic')
        #else: plot = plt.imshow(np.transpose(im_fft.real),cmap='seismic')
        #cbar = plt.colorbar(plot)
        
        #im_new = fftpack.ifftn(im_fft).real
        im_new = fftpack.ifftn(temp).real
        
        plt.figure()
        if self.axis==3: plot = plt.imshow(np.transpose(im_new[0,:,:]),cmap='seismic')
        else: plot = plt.imshow(np.transpose(im_new),cmap='seismic')
        cbar = plt.colorbar(plot)

        return im_new
    
    
    def boxfilt(self, im, dim):
        #Box filter
        
        #plt.figure()
        #if self.axis==3: plot = plt.imshow(np.transpose(im[0,:,:]),cmap='seismic')
        #else: plot = plt.imshow(np.transpose(im),cmap='seismic')
        #cbar = plt.colorbar(plot)
        
        if self.axis==2: im_new = cv2.boxFilter(im, ddepth=-1,ksize=(self.fm,self.fm))
        #elif self.axis==3: im_new = cv2.boxFilter(im, ddepth=-1,ksize=(self.fm,self.fm))
        else: print('axis error during boxfilt')
        
        #plt.figure()
        #if self.axis==3: 
        #    for i in range(100):
        #        plt.figure()
        #        plot = plt.imshow(np.transpose(im_new[0,:,:]),cmap='seismic')
        #else: plot = plt.imshow(np.transpose(im_new),cmap='seismic')
        #cbar = plt.colorbar(plot)
        
        return im_new
    
    def gauss(self, im, dim, ):
        from scipy import ndimage
        return ndimage.gaussian_filter(im,self.sigma)
    
    
    def filtvar(self, var, dim):
        #---> FIX dims! Fixed?
        if np.shape(var)[0]>0:
            varfilt = np.zeros(np.shape(var))
            for i in range(np.shape(var)[0]):
                print('start', i)
                varfilt[i] = np.array([self.filt(var[i], dim)])
                #varfilt = np.array([self.filt(var[i], dim) for i in range(np.shape(var)[0])])
                #var[i] = 0
                print(i)
        else:
            varfilt = np.array(self.filt(var, dim))
        return varfilt
    
    
    def ImportData(self, var, path=None):
        if path==None: 
            path='%s/%s/%d/%s%d%s_t%.4f.dat'%(self.path, self.dataset, self.max_dim, 
                                                self.dataset,self.max_dim,var,self.t)
        else:
            path=path
            
        #import 2D data from .dat
        print(path)
        
        try:
            if path.endswith('.dat'): 
                content = np.genfromtxt(path)
                content = np.reshape(content,(self.max_dim,self.max_dim,np.shape(content)[-1]))
            elif path.endswith('.h5'): 
                inh5 = h5.File(path, 'r')
                content=inh5['%s00000'%var]
            else: sys.exit('I am lost :( ') 
        except:
            sys.exit('Error: Make sure the path is in the format:\n'
                      '"path/dim/[dataset][dim][var]_t[time].dat" with [time] in %.4f format\n'
                      'Or edit ImportData function in functions.py')
        print(np.shape(content))
        if len(np.shape(content))>1: content = np.moveaxis(content,-1,0)
        else: content = [content]
        
        return content

    
    def ReduceDim(self, content):
        step = int((self.max_dim/self.dim))
        
        #content = np.reshape(content,(np.shape(content)[0], self.max_dim,self.max_dim))
        print('step', np.shape(content), step, self.dim, self.max_dim, type(content))        
        
        if len(np.shape(content))==self.axis: content = np.array([content]);spec = True
        else:spec=False
        
        if self.axis==2: content_new = np.zeros((np.shape(content)[0], self.dim,self.dim))
        elif self.axis==3: content_new = np.zeros((np.shape(content)[0], self.dim,self.dim,self.dim))
            
        print(np.shape(content_new), self.dim)        
        for i in range(self.dim):
            for j in range(self.dim):
                if self.axis==2: content_new[:,i,j] = content[:,int(step/2)+i*step,int(step/2)+j*step]
                elif self.axis==3:
                    for k in range(self.dim):
                        content_new[:,i,j,k] = content[:, int(step/2+i*step), int(step/2+j*step), int(step/2+k*step)]
                
        if spec: content_new = content_new[0]

        content_new = np.reshape(content_new, (np.shape(content)[0], self.dim**self.axis))
        return content_new
    
    
    def tensor(self, u, dim):
        #calculates stress tensor components
        #form = np.shape(u)
        tn = np.zeros(np.shape(u))#(3,form[0],form[1],form[2]))
        for i in range(1):
            for j in range(3):
                tn[j] = self.filt(u[i]*u[j], dim)-self.filt(u[i], dim)*self.filt(u[j], dim)    
        return tn
    
    
    def load(self, *args):   
        #load data into an array to feed to the model.fit
        vals = np.zeros(max(np.shape(args[0]))**2)
        for arg in args:
            if len(np.shape(arg))==3:
                for temp in arg:
                    vals = np.dstack((vals,temp.flatten()))
            else:
                vals = np.dstack((vals,arg.flatten()))
        return vals[0,:,1:]
        
    def defaults(self, var, kwargs):
        default = {'vmin':min(var.flatten()), 
           'vmax':max(var.flatten()), 
           'cmap':'viridis', 
           'name':''}
     
        for key in default:
            if key not in kwargs:
                kwargs[key]=default[key]
        return kwargs
        
    def plotim(self, var, **kwargs): 
        #Makes a suite of subplots with adjusted colorbars
        kwargs = self.defaults(var, kwargs)

            
        if len(np.shape(var))==3:
            fig = plt.figure(figsize = (17, 4))
            for i in range(len(var)):
                a = fig.add_subplot(1,len(var),i+1)
                im = a.imshow(var[i], cmap=kwargs['cmap'], vmin=kwargs['vmin'], vmax = kwargs['vmax'])
                plt.colorbar(im)
                plt.title('%s %d'%(kwargs['name'], i+1), fontsize=13)
        else:
            plt.figure(figsize = (6, 6))
            plt.imshow(var, cmap=kwargs['cmap'], vmin=kwargs['vmin'], vmax = kwargs['vmax'])
            plt.colorbar()
            plt.title('%s'%(kwargs['name']), fontsize=15)
            
        return kwargs['vmin'], kwargs['vmax']
    
    
    def prediction(self, var, pred,**kwargs):  
        #reformats prediction; topology comparison plots
        
        kwargs = self.defaults(var, kwargs)
        
        if self.axis==2: pred = np.reshape(pred, (self.dim,self.dim)); var = np.reshape(var, (self.dim,self.dim)) 
        if self.axis==3: 
            print('3d plotting prediction!')
            pred = np.reshape(pred, (self.dim,self.dim,self.dim))[:,:,0]
            var = np.reshape(var, (self.dim,self.dim,self.dim))[:,:,0]
        
        fig = plt.figure(figsize = (8, 6))
        #a = fig.add_subplot(121)
        im = plt.imshow(var, cmap=kwargs['cmap'], vmin=kwargs['vmin'], vmax = kwargs['vmax'])
        plt.colorbar(im).ax.tick_params(labelsize=14)
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        plt.title('Original %s'%kwargs['name'],fontsize = 17)
        if self.savepath:
            fig.tight_layout()
            fig.savefig(self.savepath+'orig.png')
        
        
        fig = plt.figure(figsize = (8, 6))
        #a = fig.add_subplot(122)
        im = plt.imshow(pred, cmap=kwargs['cmap'], vmin=kwargs['vmin'], vmax = kwargs['vmax'])
        plt.colorbar(im).ax.tick_params(labelsize=14)
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        plt.title('Predicted %s'%kwargs['name'],fontsize = 17)
        if self.savepath:
            fig.tight_layout()
            fig.savefig(self.savepath+'pred.png')
        
        return pred
    
    
    def pdf(self, var, bins=100):
        #produces a pdf plot
        
        if len(var)==2: names = ['original', 'predicted']
        else: 
            names = range(len(var))
            names = [str(i) for i in names]
        
        plt.figure('pdf', figsize = (6, 6))

        if len(np.shape(var))>1:
            val = np.zeros((np.shape(var)[0],self.dim**self.axis))    
            for i in range(len(var)):
                val[i] = np.sort(var[i])
                count = 0
                for b in range(len(val[i])):
                    if val[i][b] > 0.012:
                        count+=1
                print('a number of outliers', count)
                
                plt.hist(val[i], bins=bins, lw=3, normed=True, histtype='step', label=names[i])
                
        else:
            plt.hist(var.flatten(), bins=100, lw=3, normed=True, histtype='step')
        plt.yscale('log')
        plt.legend(fontsize=14)
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        plt.xlabel('data values', fontsize=15)
        plt.ylabel('PDF', fontsize=15)
        if self.savepath:
            plt.tight_layout()
            plt.savefig(self.savepath+'pdf.png')
        return

        
    def cdf(self, var):
        from scipy.stats import ks_2samp
        from scipy.interpolate import interp1d

        #produces a cdf plot
        if len(var)==2: names = ['original', 'predicted']
        else: 
            names = range(len(var))
            names = [str(i) for i in names]
        
        plt.figure('cdf', figsize=(6,6))
        func = []
        val = np.zeros((np.shape(var)[0],self.dim**self.axis))    
        for i in range(len(var)):
            val[i] = np.sort(var[i])

            #cdf calculation via linear interpolation
            length = len(val[i])
            yvals = np.linspace(0,length-1, length)/length
            plt.plot(val[i], yvals, label=names[i])
            func.append(interp1d(val[i], yvals))                                                            

            if i==1:
                ks_stat, pvalue = ks_2samp(val[0], val[1])
                minima = max([min(val[0]), min(val[1])])
                maxima = min([max(val[0]), max(val[1])])

                xtest = np.linspace(minima, maxima, length*10)
                D = abs(func[0](xtest)-func[1](xtest))
                Dmax = max(D)
                Dpos = xtest[np.argmax(D)]
                plt.axvline(x=Dpos, linewidth=1, color='tab:red', linestyle='--')

                txt = 'pvalue = %.3e\nks_stat = %.3e\nks_line = %.3e\nline_pos = %.3e'%(pvalue,ks_stat, Dmax, Dpos)
                plt.figtext(0.15, 0.6, txt, fontsize=14)        
        
        plt.legend(fontsize=13)
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        plt.xlabel('data values', fontsize=15)
        plt.ylabel('CDF', fontsize=15)
        plt.title('t = %.4f'%self.t, fontsize=17)
        if self.savepath:
            plt.tight_layout()
            plt.savefig(self.savepath+'cdf.png')
        plt.show()
        plt.clf()

        try:
            return pvalue, ks_stat, Dmax, Dpos
        except:
            return 100, 100, 100, 100
        
        
    def parity(self, var, pred):
        from scipy.stats import kde

        #parity plots
        x = var.flatten()
        y = pred.flatten()
        vmin = min(var.flatten())*1.1
        vmax = max(var.flatten())*1.1
        vymin = min(pred.flatten())*1.1
        vymax = max(pred.flatten())*1.1
         
        # Evaluate a gaussian kde on a regular grid of nbins x over data extents
        nbins=300
        k = kde.gaussian_kde([x,y])
        xi, yi = np.mgrid[vmin:vmax:nbins*1j, vymin:vymax:nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        
        plt.figure(figsize=(7,6))
        zi = np.ma.masked_where(zi <= 1, zi)
        cmap = plt.cm.get_cmap('Oranges')

        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=cmap,norm=colors.LogNorm())
        plt.colorbar().ax.tick_params(labelsize=14)
        one = np.linspace(vmin, vmax)
        plt.plot(one,one,'--', c='tab:red', label='y=x', alpha=0.5)
        plt.xlim([vmin,vmax])
        plt.ylim([vymin,vymax])
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        if self.savepath:
            plt.tight_layout()
            plt.savefig(self.savepath+'parity.png')
            

    def hyper3d(self, model, quantity):
        from mpl_toolkits.mplot3d import Axes3D

        print('best parameters', model.best_estimator_)
        print(model.cv_results_[quantity])
        for keys,values in model.cv_results_.items(): print(keys)

        length = len(model.cv_results_['params'])
        alpha = np.zeros(length)
        gamma = np.zeros(length)
        result = np.zeros(length)
        
        for i in range(length):
            alpha[i] = model.cv_results_['param_alpha'][i]
            gamma[i] = model.cv_results_['param_gamma'][i]
            result[i] = model.cv_results_[quantity][i]
            
            if alpha[i] == model.best_params_['alpha'] and gamma[i] == model.best_params_['gamma']:
                besta = np.log10(alpha[i])
                bestg = np.log10(gamma[i])
                bestr = result[i]
        
        alpha = [np.log10(i) for i in alpha]
        gamma = [np.log10(i) for i in gamma]
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(alpha, gamma, result, marker = 'o', label='search')
        ax.scatter(besta, bestg, bestr, marker = 'o', s=30, color='r', label='best')

        if len(alpha)>1:
            ax.plot_trisurf(alpha, gamma, result, alpha = 0.5)
            ax.set_xlabel('log(alpha)')
            ax.set_ylabel('log(gamma)')
            ax.set_zlabel(quantity)
        if self.savepath:
            fig.tight_layout()
            fig.savefig(self.savepath+'3Dpars.png')
            

    def plot_pars(self, ax, scores, x, index, yname, xname, ticks):
        plt.plot(x, scores[:,index], 'o-', label='train')
        plt.plot(x, scores[:,index+1], 'o-', label='test')
        if ticks==True: 
            plt.gca().set_xticks(x)
        else:
            ax.get_xaxis().set_major_locator(MaxNLocator(integer=True)) 
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.xlabel(xname, fontsize=15)
        plt.ylabel(yname, fontsize=15)
        plt.title(r'$\tau_{1%d}$'%self.tnc, fontsize=17)
        #if 'Mean' in yname:
        #    plt.ylim([0.000-0.001*0.2,0.0055])
        #else:
        #    plt.ylim([0.35,1.02])
    
    
    def plot_scores(self, x=None, names=None, xtitle='Setup', ticks=False, size=[12,4]):
        if self.savepath==None: sys.exit("Path is not indicated: set 'functions.savepath' to a path")
        scores = np.genfromtxt(self.savepath+'scores.dat', skip_header=1)

        if len(np.shape(scores))==1: scores=np.array([scores])
        scores = scores[:,1:]
        
        length=len(scores)

        if x!=None:pass 
        else: x = np.linspace(1, length, length)
        
        
        fig = plt.figure(figsize = (size[0], size[1]))
        ax = fig.add_subplot(1,2,1)
        self.plot_pars(ax, scores, x, 0, 'Mean_Absolute_Error', xtitle, ticks)
        ax = fig.add_subplot(1,2,2)
        self.plot_pars(ax, scores, x, 2, 'Explained_Variance_Score', xtitle, ticks)

        if names!=None:
            setups = ''
            for i in range(length): setups=setups+'[%s] '%(i)+names[i]+'\n'
            #for i in range(length): setups+'[%s]'%(i)        
            plt.figtext(0.4, 0, setups, fontsize=15, horizontalalignment='left', verticalalignment='top')
        plt.tight_layout()
        plt.savefig(self.savepath+'scores.png')
        

    def Check_Directories(self):
        #create appropriate directories if those don't exist
        
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        else:
            pass
        return
