#from Sapsan import Sapsan
import sys
import h5py as h5
import numpy as np
import time
import torch
import torch.utils.data as Tdata
from skimage.util.shape import view_as_blocks

from filters import Filters

class Data:

    def get_features(self, params, label, train=False):
        #prepare the data into the right format
        vals = None 
        
        print('Features to use: ',params)

        for tnum in range(len(self.ttrain)):
            self.first_var = True

            #import each variable
            print(type(params), type(label))
            
            #make sure that parameters and their respective labels are in a form of a list
            try: iter_test = iter(params)
            except: params=[params]
            if type(params)==str: params=[params]
            
            try: iter_test = iter(label); 
            except: label=[label]
            if type(label)==str: params=[label]
            
            for i in range(len(params)):
                if label[0]!=None: j=i
                else: j=0

                vals = self.import_var(vals, params[i], label[j], tnum, train)

            if tnum==0: size_vals=vals.shape[0]
        
        print('Size_Vals',size_vals)

        #flatten and concatenate everything to prepare for training/prediction 
        new_vals = np.zeros(self.dim**self.axis*len(self.ttrain))    
        for i in range(size_vals):
            if len(self.ttrain)>1:
                for n in range(1,len(self.ttrain)):
                    if n==1: new = np.concatenate((vals[i].flatten(), vals[i+n*size_vals].flatten()))
                    else: new = np.concatenate((new, vals[i+n*size_vals].flatten()))
            else:
                new = vals[i].flatten()
            new_vals = np.vstack((new_vals, new))
            
        new_vals = np.delete(new_vals,0, 0)
        new_vals = np.moveaxis(new_vals,-1,0)
        
        return new_vals


    def import_var(self, vals, var, label, tnum, train):
        if ('filt' in var) or (var[0]=='d' and len(var)>1):
            print('Will filter: ', var)
            check = var.replace('filt', '')
        elif 'calc' in var:
            print('Calculating Velocity Tensor')
            try: u = self.import_var(None, 'u', None, tnum, train)
            except: sys.exit("Provide 'u' to calculate the tensor components")
            x = self.tensor(u, self.dim)
            check = None
        else:
            check = var

        #check if the requested quantity can be imported directly

        try:
            if check==None: pass
            else: temp_vals = self.ImportData(tnum, check, label)
        except:
            #'ignoring said variable'
            sys.exit("Cannot import variable '%s'; it is being ignored"%check)

        tgr=time.time()
        vals = self.check_add(vals, temp_vals, var, tnum, train)
        print('time spent for filtering and reducing',time.time()-tgr)
        return vals


    def check_add(self, vals, temp_vals, var, tnum, train):
        if 'filt' in var: temp_vals = self.filtvar(temp_vals, self.max_dim)
        
        if self.axis==2 and self.from3D==True: temp_vals = temp_vals[:,:,:,0]
        temp_vals = self.ReduceDim(self.ttrain[tnum],temp_vals)
        try: vals = np.concatenate((vals, temp_vals))
        except: 
            vals = np.zeros((1,self.dim**self.axis))
            print(self.dim, self.axis)
            print('SHAPES', np.shape(vals), np.shape(temp_vals))
            vals = np.concatenate((vals, temp_vals))
            vals = np.delete(vals,0,0)

        return vals        
    
    def ImportData(self, tnum, var, varLabel=None):

        print(self.ttrain)
        print(tnum)
        print(self.dt)
        n_checkpoint=self.dt*int(self.ttrain[tnum])

        if self.path==None:
            inpath = '%s.h5'%var  
        elif '@' in self.path:
            i = self.path.find('@')
            try: 
                n = int(self.path[i+1])
                inpath = self.path.replace(self.path[i:i+2], '%0*d' % (n, n_checkpoint))
                
            except:   
                n = int(self.path[i+2])
                inpath = self.path.replace(self.path[i:i+3], '%.*f' % (n, n_checkpoint))
                
        else: inpath = self.path
        
        inpath = inpath+'%s.%s'%(var, self.dataType)

        print('Importing file: %s'%inpath)    
        
        if inpath.endswith('.dat') or inpath.endswith('.txt'):
            content = np.genfromtxt(inpath)
            content = np.reshape(content,(self.max_dim,self.max_dim,np.shape(content)[-1]))
            
        elif inpath.endswith('.h5') or inpath.endswith('.hdf5'): 
            inh5 = h5.File(inpath, 'r')
            print('inh5 imported')
            print('HDF5 file contains the following:', inh5.keys())
            if varLabel==None: varLabel = list(inh5.keys())[-1]

            print('Importing the dataset: ', varLabel)
            
            content=inh5[varLabel]
            
        else: sys.exit('Incorrect file format. Please use acceptable .dat, .txt, .h5, or .hdf5') 

        print(np.shape(content))
        if len(np.shape(content))>1: content = np.moveaxis(content,-1,0)
        else: content = [content]
        
        return content 
    
    def tensor(self, u, dim):
        #calculates stress tensor components
        self.filt = getattr(Filters, self.filtname)

        if self.axis==2: u = np.reshape(u, (3, self.dim, self.dim))
        elif self.axis==3: u = np.reshape(u, (3, self.dim, self.dim, self.dim))
        
        tn = np.zeros(np.shape(u))
        for i in range(1):
            for j in range(3):
                tn[j] = self.filt(self, u[i]*u[j], dim)-self.filt(self, u[i], dim)*self.filt(self, u[j], dim)    
        return tn
    
    def ReduceDim(self, n_checkpoint, content):        
        if self.step==None: self.step = int(self.max_dim/self.dim)
        if self.step>int(self.max_dim/self.dim): 
            sys.exit('Step size given is larger then dimMax/dim; maximum step size is %d'%int(self.max_dim/self.dim))
        
        if len(np.shape(content))==self.axis: content = np.array([content]);spec = True
        else:spec=False
        
        if self.axis==2: content_new = np.zeros((np.shape(content)[0], self.dim,self.dim))
        elif self.axis==3: content_new = np.zeros((np.shape(content)[0], self.dim,self.dim,self.dim))  
        
        loc = (n_checkpoint-int(n_checkpoint))*100

        #>>>Warn about what's allowed for 2D and 3D corners<<<
        crn = int(str(loc)[:1])
        if crn==0: corner = [1,1,1] 
        elif crn==1: corner = [-1,1,1]
        elif crn==2: corner = [-1,-1,1] 
        elif crn==3: corner = [1,-1,1]
        elif crn==4: corner = [-1,-1,-1]
        elif crn==5: corner = [1,-1,-1]
        elif crn==6: corner = [-1,1,-1]
        elif crn==7: corner = [1,1,-1]
        else: sys.exit('%s is an invalid sampling location; ex: .1, .22, or .03'%loc)
        
        sh = int(str(loc)[-1:])
        if sh==0: shift = [0,0,0] 
        elif sh==1: shift = [self.step*self.dim,0,0]
        elif sh==2: shift = [0,self.step*self.dim,0] 
        elif sh==3: shift = [0,0,self.step*self.dim]
        else: sys.exit('%s is an invalid sampling location, ex: .1, .22, or .03'%loc)
        
        print(np.shape(content_new), self.dim)        
        for i in range(self.dim):
            for j in range(self.dim):
                if self.axis==2: 
                    content_new[:,i,j] = content[:, corner[0]*i*self.step+shift[0], corner[1]*j*self.step+shift[1]]
                elif self.axis==3:
                    for k in range(self.dim):
                        content_new[:,i,j,k] = content[:, corner[0]*i*self.step+shift[0], 
                                                        corner[1]*j*self.step+shift[1], corner[2]*k*self.step+shift[2]]
                
        if spec: content_new = content_new[0]
        
        content_new = np.reshape(content_new, (np.shape(content)[0], self.dim**self.axis))
        return content_new
   

    def format_data_to_device(self, dataset, vals, var):
        X_3d = vals.reshape(self.dim*len(self.ttrain),self.dim,self.dim,vals.shape[-1])
        y_3d = var.reshape(self.dim*len(self.ttrain),self.dim,self.dim)

        print('SHAPES_to_device', X_3d.shape, y_3d.shape)

        if dataset == 'train':            
            train_size = int(self.dim*len(self.ttrain)*self.train_fraction)
            if self.batch_size==1:
                X_3d_in, y_3d_in, block, block_size = self.form_3d_tensors(X_3d[:train_size,:,:,:], y_3d[:train_size,:,:])
            else:
                X_3d_in, y_3d_in, block, block_size = self.form_3d_tensors(X_3d, y_3d)
                X_3d_in, y_3d_in = (X_3d_in[:int(X_3d_in.shape[0]*self.train_fraction)],
                                    y_3d_in[:int(y_3d_in.shape[0]*self.train_fraction)])
            
            if len(self.ttrain)>1: y_3d_in = y_3d_in[:,int(self.cube_size**3*len(self.ttrain)*self.train_fraction):]
            X_3d_in = torch.from_numpy(X_3d_in).float()
            y_3d_in = torch.from_numpy(y_3d_in).float()

            print('Shapes of train tensors', np.shape(X_3d_in), np.shape(y_3d_in))
            
            train = Tdata.DataLoader(dataset = Tdata.TensorDataset(X_3d_in, y_3d_in),
                                        batch_size = self.batch_size,
                                        shuffle = False,
                                        num_workers = 4)
            
            
            if self.train_fraction == 1: valid_size = 0
            else: valid_size = train_size
                
            if self.batch_size==1:            
                X_3d_in, y_3d_in, block, block_size = self.form_3d_tensors(X_3d[valid_size:,:,:,:], y_3d[valid_size:,:,:])
            else:
                X_3d_in, y_3d_in, block, block_size = self.form_3d_tensors(X_3d, y_3d)
                X_3d_in, y_3d_in = (X_3d_in[int(X_3d_in.shape[0]*self.train_fraction):], 
                                    y_3d_in[int(y_3d_in.shape[0]*self.train_fraction):])
            
            if len(self.ttrain)>1: y_3d_in = y_3d_in[:,int(self.cube_size**3*len(self.ttrain)*self.train_fraction):]
                
            X_3d_in = torch.from_numpy(X_3d_in).float()
            y_3d_in = torch.from_numpy(y_3d_in).float()

            print('Shapes of valid tensors', np.shape(X_3d_in), np.shape(y_3d_in))
            
            valid = Tdata.DataLoader(dataset = Tdata.TensorDataset(X_3d_in, y_3d_in),
                                        batch_size = self.batch_size,
                                        shuffle = False,
                                        num_workers = 4)            
            
            loaders = {dataset: train, "valid": valid}        
        
        elif dataset == 'test':
            train_size = int(self.dim*len(self.ttrain)*self.train_fraction)
            
            if self.batch_size==1: 
                X_3d_in, y_3d_in, block, block_size = self.form_3d_tensors(X_3d[:train_size,:,:,:], y_3d[:train_size,:,:])
            else:
                X_3d_in, y_3d_in, block, block_size = self.form_3d_tensors(X_3d, y_3d)
                X_3d_in, y_3d_in = (X_3d_in[int(X_3d_in.shape[0]*self.train_fraction):], 
                                    y_3d_in[int(y_3d_in.shape[0]*self.train_fraction):])
            
            if len(self.ttrain)>1: 
                y_3d_in = y_3d_in[:,int(self.cube_size**3*len(self.ttrain)*self.train_fraction):]
                block_size[0] = int(block_size[0]/len(self.ttrain))
                block[0] = int(block[0]*self.train_fraction)
            
            X_3d_in = torch.from_numpy(X_3d_in).float()
            y_3d_in = torch.from_numpy(y_3d_in).float()

            print('Shapes of test tensors', np.shape(X_3d_in), np.shape(y_3d_in))
            
            test = Tdata.DataLoader(dataset = Tdata.TensorDataset(X_3d_in, y_3d_in),
                                        batch_size = self.batch_size,
                                        shuffle = True,
                                        num_workers = 4)
            loaders = {dataset: test}

        return loaders, block, block_size


    def form_3d_tensors(self, X, y):        
        """ Form cubes size CUBE_SIZE**3 for 3d convolutions """
        
        import matplotlib.pyplot as plt
        
        train_X = np.moveaxis(X, -1, 0)
        train_y = y
        
        #combine nx with cubex?
        batch = self.batch_size
        block_size = np.ones(3, int)
        while batch>=2:
            block_size[0] *= 2; batch /= 2
            if batch>=2: block_size[1] *= 2; batch /= 2
            if batch>=2: block_size[2] *= 2; batch /= 2
        for i in range(len(block_size)): block_size[i] = int(X.shape[i]/block_size[i])
        
        print('form_3d_tensors, pre_blocks', np.shape(train_X), np.shape(train_y))
        train_X = view_as_blocks(train_X,(train_X.shape[0], block_size[0], block_size[1], block_size[2]))[0]
        train_y = view_as_blocks(train_y,(block_size[0], block_size[1], block_size[2]))

        print('form_3d_tensors', np.shape(train_X), np.shape(train_y))
        
        block = [train_y.shape[0], train_y.shape[1], train_y.shape[2]]
        train_X = np.reshape(train_X, (train_X.shape[0]*train_X.shape[1]*train_X.shape[2], 
                                       train_X.shape[3], block_size[0], block_size[1], block_size[2]))
        temp = train_y
        train_y = np.reshape(train_y, (train_y.shape[0]*train_y.shape[1]*train_y.shape[2], 
                                       block_size[0]*block_size[1]*block_size[2]))
        
        return np.array(train_X), np.array(train_y), block, block_size


    def filtvar(self, var, dim):
        self.filt = getattr(Filters, self.filtname)
        
        if np.shape(var)[0]>1:
            varfilt = np.zeros(np.shape(var))
            for i in range(np.shape(var)[0]):
                varfilt[i] = np.array([self.filt(self, var[i], dim)])
                var[i] = 0
        else:
            varfilt = np.array(self.filt(self, var, dim))
        return varfilt
 

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
