import numpy as np
from sklearn.kernel_ridge import KernelRidge
import time
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error

import torch
from torch.autograd import Variable
import torch.nn.functional as TFunc
import torch.utils.data as Tdata

from catalyst.dl.callbacks import EarlyStoppingCallback
from catalyst.dl import SupervisedRunner

import mlflow

from DataAnalysis import Data
from ResultAnalysis import Results
from parameters import parameters
from CNN import SpacialConvolutionsModel

import h5py as h5

class Sapsan:

    def __init__(self, pars):        
        self.__dict__ = parameters.LoadParameters(pars)
        self.data = Data()
        self.results = Results()
        for key, value in self.__dict__.items(): 
                self.data.__dict__[key]=value
                self.results.__dict__[key]=value
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.mlflow_url = 'http://localhost:9999'
    
    def fit(self):
        #the functions trains and outputs the model
        #allpars = parameters.LoadParameters(pars)
        
        mlflow.set_tracking_uri(self.mlflow_url)
                
        exp_id = mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(experiment_id=exp_id, nested=True, run_name=self.name):
            for key, value in self.__dict__.items(): 
                self.data.__dict__[key]=value
                self.results.__dict__[key]=value
                if key not in ['experiment_name','name','mlflow_url','data','results','device']: 
                    mlflow.log_param(key,value)
            
            #import features of the training set
            vals = self.data.get_features(self.parameters, self.parLabel, train=True)
            target = self.data.get_features(self.target, self.targetLabel, train=True)
            
            print('Imported train values shape: ', np.shape(vals))
            print('Imported target values shape: ', np.shape(target))
            
            if len(np.shape(target))>1: var = target[:,self.targetComp]
            else: var = target
        

            if self.method=='krr':
                #select a KRR model with appropriate hyperparameters
                if self.alpha==None or self.gamma==None:
                    model = KernelRidge(kernel='rbf')
                else:
                    model = KernelRidge(kernel='rbf', alpha = self.alpha, gamma = self.gamma)

                #extracting specific variables for the training dataset (DS) --->needs a fix for correct vals order
                u = vals[:,:3]
                du = vals[:,3:12]

                X_train = vals; y_train = var
                
                #perform actual fitting
                start_train = time.time()
                model.fit(X_train, y_train)
                print('Training time: %.1f'%(time.time()-start_train))

                pred = model.predict(X_train)

            elif self.method=='cnn':
                loaders, block, block_size = self.data.format_data_to_device("train",vals, var)
                
                train_size = int(self.dim*len(self.ttrain)*self.train_fraction)
                
                if self.batch_size==1: D_out = int(self.cube_size**3*self.train_fraction)
                else: D_out = int(self.cube_size**3)
                    
                model = SpacialConvolutionsModel(vals.shape[-1], D_out)
                model.to(self.device)

                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                loss_func = torch.nn.MSELoss() #torch.nn.SmoothL1Loss()
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer,
                            patience = 3,
                            min_lr = 1e-5)

                runner = SupervisedRunner()

                #perform actual fitting
                start_train = time.time()
                
                runner.train(model=model,
                            criterion = loss_func,
                            optimizer = optimizer,
                            scheduler = scheduler,
                            loaders = loaders,
                            logdir = self.savepath,
                            num_epochs = self.n_epochs,
                            callbacks = [EarlyStoppingCallback(patience=10, min_delta = 1e-5)],
                            verbose = False,
                            check = False)

                print('Training time: %.1f'%(time.time()-start_train))
                
                todevice = loaders['train'].dataset.tensors[0].to(self.device)
                
                print('train todevice size', np.shape(todevice))
                pred = model(todevice).cpu().data.numpy().reshape(-1)
                y_train = loaders['train'].dataset.tensors[1].data.numpy().reshape(-1)

            #>>>add randomly splitting data<<<
            
            #print out the scoring
            variance = explained_variance_score(y_train, pred)
            abserr = mean_absolute_error(y_train, pred)
            
            mlflow.log_metric('variance', variance)
            mlflow.log_metric('mean_abs_err', abserr)

            self.step = None
            
            print('Against itself, variance=%.4f and abserr=%.4f'%(variance, abserr))
                
        return model
    
    
    def test(self, model, ttest):    
        
        mlflow.set_tracking_uri(self.mlflow_url)
        exp_id = mlflow.set_experiment(self.experiment_name+' test')
        with mlflow.start_run(experiment_id=exp_id, nested=True, run_name=self.name):
            for key, value in self.__dict__.items():
                self.data.__dict__[key]=value
                self.results.__dict__[key]=value
                if key not in ['experiment_name','name','mlflow_url','data','results','device']: 
                    mlflow.log_param(key,value)
            #>>>>Check if sps.pars can be updated after training
            
            #loop over all the test timesteps to predict
            #for i in range(len(ttest)):
            for i in range(1):
                t1 = time.time() #begin timer

                #get vars of the test timestep
                print('>>>TEST TIME<<<', i, ttest, ttest[i], self.ttrain)
                self.data.ttrain = ttest

                #>>>Why self.step gets reset to 1??<<<
                #self.dim = self.max_dim; self.step = 1 #int(self.max_dim/2)
                
                test_vals = self.data.get_features(self.parameters, self.parLabel)
                target_vals = self.data.get_features(self.target, self.targetLabel)

                if len(np.shape(target_vals))>1: target = target_vals[:,self.targetComp]
                else: target = target_vals

                if self.method=='krr':
                    pred_test = model.predict(test_vals)
                elif self.method=='cnn':
                    loaders, block, block_size = self.data.format_data_to_device("test", test_vals, target)
                    todevice = loaders['test'].dataset.tensors[0].to(self.device)
                    print('------')
                    print('shape to device, ', np.shape(todevice), np.shape(model(todevice).cpu().data.numpy()))
                    pred_test = model(todevice).cpu().data.numpy().reshape(-1)

                    print('-->>> Loaders[test] shape ',np.shape(loaders['test'].dataset.tensors[1].data.numpy()))
                    target = loaders['test'].dataset.tensors[1].data.numpy().reshape(-1)

                print('SHAPES target, pred_test', np.shape(target), np.shape(pred_test))
                variance_test = explained_variance_score(target, pred_test)
                abserr_test = mean_absolute_error(target, pred_test)          
                mlflow.log_metric('variance', variance_test)
                mlflow.log_metric('mean_abs_err', abserr_test)

                #proper output formatting if a sequence of test timesteps is given
                if len(ttest)!=1: self.results.savepath = self.savepath+'%ddt/'%self.ttrain[0]

                self.results.Check_Directories()

                self.results.ttest = self.dt*ttest[i] #>>>could be an issue?<<<

                #plot various quntities 
                self.results.prediction(target, pred_test, block, block_size, cmap='ocean', name=r'$\tau_{1%d}$'%self.targetComp)
                #self.results.output(target, pred_test)
                
                #>>>Uncomment after fixing the dims!
                #hf = h5.File('%soutput.h5'%(self.savepath), 'w')
                #a = hf.create_dataset('original', (self.dim,self.dim,self.dim))
                #b = hf.create_dataset('predict', (self.dim,self.dim,self.dim))
                
                #a[:,:,:] = np.reshape(target,(a.shape))
                #b[:,:,:] = np.reshape(pred_test,(b.shape))
                
                #hf.close()

                scores = np.zeros((len(ttest),2))
                stats = np.zeros((len(ttest),4))
                
                #target = DS(u, du, self.dim)
                #stats[i] = Result.pdf(np.stack((target,result, target[0,1])))
                self.results.pdf(np.stack((target,pred_test)))
                stats[i] = self.results.cdf(np.stack((target,pred_test)))
                #Results.parity(target, pred_test)

                scores[i] = [abserr_test, variance_test]
                t2 = time.time()
                print('Retrieval time: ', (t2-t1))
                print('OUTPUT SHAPES target, pred_test', np.shape(target), np.shape(pred_test))

        #return scores, stats
        return target, pred_test
