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

import time
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np

from sapsan.core.models import Experiment, ExperimentBackend, Estimator
from sapsan.utils.plot import pdf_plot, slice_of_cube
from sapsan.utils.shapes import combine_cubes


class EvaluateFlatten(Experiment):
    def __init__(self,
                 name: str,
                 backend: ExperimentBackend,
                 model: Estimator,
                 inputs: np.ndarray,
                 targets: np.ndarray,
                 checkpoint_data_size: int,
                 checkpoints: List[float],
                 axis: int,
                 cmap: str = 'ocean'):
        super().__init__(name, backend)
        self.model = model
        self.inputs = inputs
        self.targets = targets
        self.n_output_channels = targets.shape[1]
        self.experiment_metrics = dict()
        self.checkpoint_data_size = checkpoint_data_size
        self.checkpoints = checkpoints
        self.axis = axis
        self.cmap = cmap
        self.artifacts = []

    def get_metrics(self) -> Dict[str, float]:
        return self.experiment_metrics

    def get_parameters(self) -> Dict[str, str]:
        return {
            "n_output_channels": str(self.n_output_channels)
        }

    def get_artifacts(self) -> List[str]:
        # TODO:
        return []

    def run(self) -> dict:
        start = time.time()

        pred = self.model.predict(self.inputs)

        plot = pdf_plot([pred, self.targets], names=['prediction', 'targets'])
        plt.show()

        n_entries = len(self.checkpoints)

        if self.axis == 3:
            cube_shape = (n_entries, 1, self.checkpoint_data_size,
                          self.checkpoint_data_size, self.checkpoint_data_size)
            pred_cube = pred.reshape(cube_shape)
            target_cube = self.targets.reshape(cube_shape)

            pred_slice = slice_of_cube(pred_cube[0])
            target_slice = slice_of_cube(target_cube[0])
        
        if self.axis == 2: 
            cube_shape = (n_entries, 1, self.checkpoint_data_size, self.checkpoint_data_size)
            pred_cube = pred.reshape(cube_shape)
            target_cube = self.targets.reshape(cube_shape)

            pred_slice = slice_of_cube(pred_cube)
            target_slice = slice_of_cube(target_cube)
        
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
        plt.savefig("./slice.jpg")
        self.artifacts.append("./slice.jpg")
        plt.show()

        end = time.time()

        runtime = end - start

        for metric, value in self.get_metrics().items():
            self.backend.log_metric(metric, value)

        for param, value in self.get_parameters().items():
            self.backend.log_parameter(param, value)

        # TODO: save image from plot and log artifact

        self.backend.log_metric("runtime", runtime)

        return {
            'runtime': runtime
        }
