"""
Example:
evaluation_experiment = Evaluate(backend=tracking_backend,
                                 model=training_experiment.model,
                                 data_parameters = data_loader)

cubes = evaluation_experiment.run()
"""

import os
import time
from typing import List, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from sapsan.core.models import Experiment, ExperimentBackend, Estimator
from sapsan.lib.backends.fake import FakeBackend
from sapsan.utils.plot import pdf_plot, cdf_plot, slice_plot, line_plot, plot_params
from sapsan.utils.shapes import combine_data, slice_of_cube

class Evaluate(Experiment):
    def __init__(self,
                 model: Estimator,
                 data_parameters,
                 backend = FakeBackend(),
                 cmap: str = 'viridis',
                 run_name: str = 'evaluate',
                 **kwargs):
        self.model = model
        self.backend = backend
        self.experiment_metrics = dict()
        self.data_parameters = data_parameters
        self.input_size = self.data_parameters.input_size
        self.batch_size = self.data_parameters.batch_size
        self.batch_num = self.data_parameters.batch_num
        self.flat = self.data_parameters.flat
        self.cmap = cmap
        self.axis = len(self.input_size)
        self.targets_given = True        
        self.run_name = run_name
        self.artifacts = []    
        self.axes_pars = ['pdf_xlim','pdf_ylim',
                          'cdf_xlim','cdf_ylim']
        self.kwargs = kwargs        
        
        if type(self.model.loaders) in [list, np.array]:
            self.inputs = self.model.loaders[0]            
            try: 
                self.targets = self.model.loaders[1]
            except: 
                self.targets_given = False
                print('Warning: no target given; only predicting...')
        else:
            try:
                self.inputs, self.targets = iter(self.model.loaders['train']).next()
                self.targets = self.targets.numpy()
            except: 
                self.inputs = iter(self.model.loaders['train']).next()[0]
                self.targets_given = False
                print('Warning: no target given; only predicting...')
        
        
    def get_metrics(self) -> Dict[str, float]:
        if 'train' in self.model.metrics():
            self.experiment_metrics['train - final epoch'] = self.model.metrics()['final epoch']
            for metric, value in self.model.metrics()['train'].items():
                if "/" in metric: metric = metric.replace("/", " over ")
                self.experiment_metrics['train - %s'%metric] = value
        else: self.experiment_metrics = {**self.experiment_metrics, **self.model.metrics()}
            
        return self.experiment_metrics

    def get_parameters(self) -> Dict[str, str]:
        return {
            **self.data_parameters.get_parameters(), 
            **self.model.config.parameters, 
            **{"n_output_channels": str(self.n_output_channels)}            
        }        

    def get_artifacts(self) -> List[str]:
        return self.artifacts

    def _cleanup(self):
        for artifact in self.artifacts:
            os.remove(artifact)
        return len(self.artifacts)

    def run(self) -> dict:
        start = time.time()

        self.run_id = self.backend.start(self.run_name, nested = True)

        predict = self.model.predict(self.inputs, self.model.config)   

        end = time.time()
        runtime = end - start
        self.backend.log_metric("eval - runtime", runtime)

        #determine n_output_channels form prediction
        if self.flat:
            self.n_output_channels = int(np.around(
                                     np.prod(predict.shape)/np.prod(self.inputs.shape[1:])
                                     )) #flat arrays don't have batches
        elif len(predict.shape)<(self.axis+2): 
            self.n_output_channels = int(np.around(
                                         np.prod(predict.shape[1:])/np.prod(self.inputs.shape[2:])
                                         ))
        else: self.n_output_channels = predict.shape[1]            

        if self.targets_given: 
            series = [self.targets, predict]
            label = ['target', 'predict']
        else: 
            series = [predict]
            label = ['predict']
        outdata = self.analytic_plots(series, label)

        if self.targets_given:
            self.experiment_metrics["eval - MSE Loss"] = np.square(np.subtract(outdata['target_restored'], 
                      outdata['predict_restored'])).mean()         

        for metric, value in self.get_metrics().items():
            self.backend.log_metric(metric, value)

        for param, value in self.get_parameters().items():
            self.backend.log_parameter(param, value)

        for artifact in self.artifacts:
            self.backend.log_artifact(artifact)

        self.backend.end()
        self._cleanup()

        print("eval - runtime: ", runtime)

        restored_series = dict()
        for key, value in outdata.items():
            if 'restored' in key: restored_series[key.split('_restored')[0]] = value

        return restored_series
    
    
    def flatten(self, predict):
        outdata = dict()    
        if self.axis == 3:            
            restored_shape = (self.n_output_channels, self.input_size[0],
                              self.input_size[1], self.input_size[2])            
        elif self.axis == 2: 
            restored_shape = (self.n_output_channels, self.input_size[0], self.input_size[1])            
        elif self.axis == 1: 
            restored_shape = (self.n_output_channels, self.input_size[0])
            
        predict_restored = predict.reshape(restored_shape)
        if self.axis == 3: predict = slice_of_cube(predict_restored)
        else: predict = predict_restored
        outdata['predict_restored'] = predict_restored                       
        outdata['predict'] = predict[0]  
                      
        if self.targets_given:            
            target_restored = self.targets.reshape(restored_shape)
            if self.axis == 3: target = slice_of_cube(target_restored)
            else: target = target_restored
            outdata['target_restored'] = target_restored                
            outdata['target'] = target[0]
        
        print("Warning: only the first target will be plotted. The full target can be found in the output['target_restored']")
        return outdata
    
    
    def split_batch(self, predict):
        outdata = dict()
        n_entries = self.inputs.shape[0]
        
        if self.axis == 3:            
            restored_shape = (n_entries, self.n_output_channels, 
                              self.batch_size[0], self.batch_size[1], self.batch_size[2])            
        elif self.axis == 2: 
            restored_shape = (n_entries, self.n_output_channels, 
                              self.batch_size[0], self.batch_size[1])            
        elif self.axis == 1: 
            restored_shape = (n_entries, self.n_output_channels, 
                              self.batch_size[0])
            
        predict_restored = predict.reshape(restored_shape)     
        if self.axis == 3: predict = slice_of_cube(combine_data(predict_restored,
                                                                self.input_size, 
                                                                self.batch_size,
                                                                self.axis))
        elif self.axis == 2: predict = combine_data(predict_restored,
                                                    self.input_size, 
                                                    self.batch_size,
                                                    self.axis)
        else: predict = predict_restored
        outdata['predict'] = predict
        outdata['predict_restored'] = predict_restored
                      
        if self.targets_given: 
            target_restored = self.targets.reshape(restored_shape)
            if self.axis == 3: target = slice_of_cube(combine_data(target_restored,
                                                                   self.input_size, 
                                                                   self.batch_size,
                                                                   self.axis))
            elif self.axis == 2: target = combine_data(target_restored,
                                                       self.input_size, 
                                                       self.batch_size,
                                                       self.axis)
            else: target = target_restored
            outdata['target'] = target
            outdata['target_restored'] = target_restored
                      
        return outdata
                      
                      
    def analytic_plots(self, series, label):
        mpl.rcParams.update(plot_params())
                      
        fig = plt.figure(figsize=(12,6), dpi=60)
        (ax1, ax2) = fig.subplots(1,2)

        pdf = pdf_plot(series, label=label, ax=ax1)
        self.set_axes_pars(pdf)
        
        cdf, ks_stat = cdf_plot(series, label=label, ax=ax2, ks=True)
        cdf = self.set_axes_pars(cdf)
        
        plt.tight_layout()
        plt.savefig("pdf_cdf.png")
        self.artifacts.append("pdf_cdf.png")                        
        
        self.experiment_metrics["eval - KS Stat"] = ks_stat
                
        predict = series[label.index('predict')]

        if self.flat: outdata = self.flatten(predict)
        else: outdata = self.split_batch(predict)
        
        plot_label = ['target','predict']
        if self.axis>1: 
            plot_series = []                        
            for key, value in outdata.items():
                if key in plot_label: 
                    plot_series.append(value)

            slices = slice_plot(plot_series, label=plot_label, cmap=self.cmap)
            
        elif self.axis==1:
            print('The data is 1D: plotting profiles...')
            profiles = line_plot([[np.arange(self.input_size[0]),outdata['target'][0,0]], 
                                  [np.arange(self.input_size[0]),outdata['predict'][0,0]]], 
                                 label=plot_label, figsize=(10,6))
            profiles.set_xlabel('index')
            profiles.set_ylabel('value')
            
        plt.tight_layout()
        plt.savefig("spatial_plot.png")
        self.artifacts.append("spatial_plot.png")
        
        return outdata
    
    def set_axes_pars(self, ax):
        for key in self.axes_pars:
            if key in self.kwargs.keys(): 
                axes_attr = getattr(ax, 'set_'+key.split('_')[-1])
                axes_attr(self.kwargs[key])                        