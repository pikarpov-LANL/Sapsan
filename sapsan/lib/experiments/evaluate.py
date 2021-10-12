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
from sapsan.utils.plot import pdf_plot, cdf_plot, slice_plot, plot_params
from sapsan.utils.shapes import combine_cubes, slice_of_cube

class Evaluate(Experiment):
    def __init__(self,
                 model: Estimator,
                 data_parameters,
                 backend = FakeBackend(),
                 cmap: str = 'viridis',
                 flat: bool = False,
                 run_name: str = 'evaluate',
                 **kwargs):
        self.model = model
        self.backend = backend
        self.experiment_metrics = dict()
        self.data_parameters = data_parameters
        self.input_size = self.data_parameters.input_size
        self.batch_size = self.data_parameters.batch_size
        self.batch_num = self.data_parameters.batch_num
        self.cmap = cmap
        self.axis = len(self.input_size)
        self.targets_given = True
        self.flat = flat
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

        pred = self.model.predict(self.inputs, self.model.config)              

        end = time.time()
        runtime = end - start
        self.backend.log_metric("eval - runtime", runtime)

        #determine n_output_channels form prediction
        if self.flat:
            self.n_output_channels = int(np.around(
                                     np.prod(pred.shape)/np.prod(self.inputs.shape[1:])
                                     )) #flat arrays don't have batches
        elif len(pred.shape)<(self.axis+2): 
            self.n_output_channels = int(np.around(
                                         np.prod(pred.shape[1:])/np.prod(self.inputs.shape[2:])
                                         ))
        else: self.n_output_channels = pred.shape[1]            

        if self.targets_given: 
            series = [self.targets, pred]
            names = ['target', 'predict']
        else: 
            series = [pred]
            names = ['predict']

        slices_cubes = self.analytic_plots(series, names)

        if self.targets_given:
            self.experiment_metrics["eval - MSE Loss"] = np.square(np.subtract(slices_cubes['target_cube'], 
                      slices_cubes['pred_cube'])).mean()         

        for metric, value in self.get_metrics().items():
            self.backend.log_metric(metric, value)

        for param, value in self.get_parameters().items():
            self.backend.log_parameter(param, value)

        for artifact in self.artifacts:
            self.backend.log_artifact(artifact)


        self.backend.end()
        self._cleanup()

        print("eval - runtime: ", runtime)

        cube_series = dict()
        for key, value in slices_cubes.items():
            if 'cube' in key: cube_series[key] = value

        return cube_series
    
    
    def flatten(self, pred):
        slices_cubes = dict()         
        if self.axis == 3:
            cube_shape = (self.n_output_channels, self.input_size[0],
                          self.input_size[1], self.input_size[2])
        if self.axis == 2: 
            cube_shape = (self.n_output_channels, self.input_size[0], self.input_size[1])
            
        pred_cube = pred.reshape(cube_shape)
        pred_slice = slice_of_cube(pred_cube)
        slices_cubes['pred_slice'] = pred_slice
        slices_cubes['pred_cube'] = pred_cube
                      
        if self.targets_given:              
            target_cube = self.targets.reshape(cube_shape)
            target_slice = slice_of_cube(target_cube)
            slices_cubes['target_slice'] = target_slice
            slices_cubes['target_cube'] = target_cube
                      
        return slices_cubes
    
    
    def split_batch(self, pred):
        slices_cubes = dict()
        n_entries = self.inputs.shape[0]
        cube_shape = (n_entries, self.n_output_channels,
                      self.batch_size[0], self.batch_size[1], self.batch_size[2])
        
        pred_cube = pred.reshape(cube_shape)        
        pred_slice = slice_of_cube(combine_cubes(pred_cube,
                                                 self.input_size, self.batch_size))
        slices_cubes['pred_slice'] = pred_slice
        slices_cubes['pred_cube'] = pred_cube
                      
        if self.targets_given: 
            target_cube = self.targets.reshape(cube_shape)
            target_slice = slice_of_cube(combine_cubes(target_cube,
                                         self.input_size, self.batch_size))
            slices_cubes['target_slice'] = target_slice
            slices_cubes['target_cube'] = target_cube
                      
        return slices_cubes
                      
                      
    def analytic_plots(self, series, names):
        mpl.rcParams.update(plot_params())
                      
        fig = plt.figure(figsize=(12,6), dpi=60)
        (ax1, ax2) = fig.subplots(1,2)

        pdf = pdf_plot(series, names=names, ax=ax1)
        self.set_axes_pars(pdf)
        
        cdf, ks_stat = cdf_plot(series, names=names, ax=ax2, ks=True)
        cdf = self.set_axes_pars(cdf)
        
        plt.tight_layout()
        plt.savefig("pdf_cdf.png")
        self.artifacts.append("pdf_cdf.png")                        
        
        self.experiment_metrics["eval - KS Stat"] = ks_stat
                
        pred = series[names.index('predict')]

        if self.flat: slices_cubes = self.flatten(pred)
        else: slices_cubes = self.split_batch(pred)
        
        slice_series = []
        slice_names = []
        for key, value in slices_cubes.items():
            if 'slice' in key: 
                slice_series.append(value)
                slice_names.append(key)
                      
        slices = slice_plot(slice_series, names=slice_names, cmap=self.cmap)
        plt.tight_layout()
        plt.savefig("slices_plot.png")
        self.artifacts.append("slices_plot.png")
                      
        return slices_cubes
    
    def set_axes_pars(self, ax):
        for key in self.axes_pars:
            if key in self.kwargs.keys(): 
                axes_attr = getattr(ax, 'set_'+key.split('_')[-1])
                axes_attr(self.kwargs[key])                        