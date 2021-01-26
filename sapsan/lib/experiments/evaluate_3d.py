"""

Example:
    experiment_name = "Training experiment"
    dataset_root_dir = "/Users/icekhan/Documents/development/myprojects/sapsan/repo/Sapsan/dataset"
    estimator = CNN3d(
        config=CNN3dConfig(n_epochs=1)
    )
    x, y = JHTDB128Dataset(path=dataset_root_dir,
                           features=['u', 'b', 'a',
                                'du0', 'du1', 'du2',
                                'db0', 'db1', 'db2',
                                'da0', 'da1', 'da2'],
                           labels=['tn'],
                           checkpoints=[0.0]).load()

    experiment = TrainingExperiment(name=experiment_name,
                                    backend=FakeBackend(experiment_name),
                                    model=estimator,
                                    inputs=x, targets=y)
    experiment.run()
"""
import os
import time
from typing import List, Dict
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from sapsan.core.models import Experiment, ExperimentBackend, Estimator
from sapsan.utils.plot import pdf_plot, cdf_plot, slice_of_cube
from sapsan.utils.shapes import combine_cubes


class Evaluate3d(Experiment):
    def __init__(self,
                 name: str,
                 backend: ExperimentBackend,
                 model: Estimator,
                 inputs: np.ndarray,
                 targets: np.ndarray,
                 data_parameters: dict,
                 cmap: str = 'ocean'
                 ):
        super().__init__(name, backend)
        self.model = model
        self.inputs = inputs
        self.targets = targets
        self.n_output_channels = targets.shape[1]
        self.experiment_metrics = dict()
        self.data_parameters = data_parameters
        self.checkpoint_data_size = self.data_parameters["checkpoint - sample to size"]
        self.grid_size = self.data_parameters["checkpoint - batch size"]

        self.cmap = cmap

        self.artifacts = []
        
        params = {'axes.labelsize': 20, 'legend.fontsize': 15, 'xtick.labelsize': 17,'ytick.labelsize': 17,
                  'axes.titlesize':20, 'axes.linewidth': 1, 'lines.linewidth': 1.5,
                  'xtick.major.width': 1,'ytick.major.width': 1,'xtick.minor.width': 1,'ytick.minor.width': 1,
                  'xtick.major.size': 4,'ytick.major.size': 4,'xtick.minor.size': 3,'ytick.minor.size': 3,
                  'axes.formatter.limits' : [-7, 7], 'text.usetex': False, 'figure.figsize': [6,6]}
        mpl.rcParams.update(params)

    def get_metrics(self) -> Dict[str, float]:
        return self.experiment_metrics

    def get_parameters(self) -> Dict[str, str]:
        return {
            **self.data_parameters, **{"n_output_channels": str(self.n_output_channels)}
        }

    def get_artifacts(self) -> List[str]:
        return self.artifacts

    def _cleanup(self):
        for artifact in self.artifacts:
            os.remove(artifact)
        return len(self.artifacts)

    def run(self) -> dict:
        start = time.time()
        
        self.backend.start('evaluate')

        pred = self.model.predict(self.inputs)
        
        end = time.time()
        runtime = end - start
        self.backend.log_metric("runtime", runtime)

        pdf = pdf_plot([pred, self.targets], names=['prediction', 'targets'])
        pdf.savefig("pdf_plot.png")
        self.artifacts.append("pdf_plot.png")

        try:
            cdf = cdf_plot([pred, self.targets], names=['prediction', 'targets'])
            cdf.savefig("cdf_plot.png")
            self.artifacts.append("cdf_plot.png")
        except Exception as e:
            logging.warning(e)

        n_entries = self.inputs.shape[0]

        cube_shape = (n_entries, self.n_output_channels,
                      self.grid_size, self.grid_size, self.grid_size)
        pred_cube = pred.reshape(cube_shape)
        target_cube = self.targets.reshape(cube_shape)

        pred_slice = slice_of_cube(combine_cubes(pred_cube,
                                                 self.checkpoint_data_size, self.grid_size))
        target_slice = slice_of_cube(combine_cubes(target_cube,
                                                   self.checkpoint_data_size, self.grid_size))

        vmin = np.amin(target_slice)
        vmax = np.amax(target_slice)
        
        fig = plt.figure(figsize = (16, 6))
        fig.add_subplot(121)
        im = plt.imshow(target_slice, cmap=self.cmap, vmin=vmin, vmax = vmax)
        plt.colorbar(im).ax.tick_params(labelsize=14)
        plt.title("Target slice")

        fig.add_subplot(122)
        im = plt.imshow(pred_slice, cmap=self.cmap, vmin=vmin, vmax = vmax)
        plt.colorbar(im).ax.tick_params(labelsize=14)
        plt.title("Predicted slice")
        plt.savefig("prediction.png")
        self.artifacts.append("prediction.png")
        plt.show()

        self.experiment_metrics["MSE Loss"] = np.square(np.subtract(target_cube, pred_cube)).mean()         

        for metric, value in self.get_metrics().items():
            self.backend.log_metric(metric, value)

        for param, value in self.get_parameters().items():
            self.backend.log_parameter(param, value)

        for artifact in self.artifacts:
            self.backend.log_artifact(artifact)
            
        self.backend.end()
        self._cleanup()
        
        print("runtime: ", runtime)
        
        return target_cube, pred_cube
