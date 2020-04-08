Sapsan
======

* [Intro](#intro)
* [Structure](#structure)
  * [Estimator](#estimator)
  * [Experiment](#experiment)
  * [Dataset](#dataset)
  * [Tracking backend](#tracking-backend) 
* [Examples](#examples)
* [Kubeflow](#kubeflow)

-----------

### Intro
TODO


---------

### Structure
Structure of project is build around few concepts making this project easier to extend to more cases.
Core abstractions are: estimator, dataset, experiment, tracking backend
Core abstraction are defined in [models.py](sapsan/core/models.py) file.

#### Estimator
General abstraction for models/algorithms.

##### Available estimators
- [KRR 1d estimator](sapsan/lib/estimator/krr/krr.py)
- [3d convolution encoder estimator](sapsan/lib/estimator/cnn/spacial_3d_encoder.py)
- [3d autoencoder estimator](sapsan/lib/estimator/cnn/spacial_autoencoder.py)

##### How to implement new estimator:

Extend `Estimator` class and implement `train`, `predict` and `metrics` methods.

```python
from sapsan.core.models import Estimator, EstimatorConfiguration


class AwesomeEstimator(Estimator):
    def __init__(self, config: EstimatorConfiguration):
        super().__init__(config)
        self.model = ... # your awesome model

    def train(self, inputs, labels):
        # fit model to labels
        return self.model

    def predict(self, inputs):
        prediction = ... # derive prediction from inputs
        return prediction

    def metrics(self):
        return {"training_time": 146, "training_avg_loss": 42}
```

#### Dataset
General abstraction for dataset/dataframes.

##### Available datasets
- [3d dataset](sapsan/lib/data/jhtdb_dataset.py)
- [1d dataset](sapsan/lib/data/flatten_dataset.py)

##### How to implement new dataset:

Extend `Dataset` class and impement `load` method.

```python
import numpy as np
from sapsan.core.models import Dataset


class RandomDataset(Dataset):
    def __init__(self, n_entries: int, n_features: int):
        self.n_entries = n_entries
        self.n_features = n_features

    def load(self):
        return np.random.random((self.n_entries, self.n_features))
```

#### Experiment

General abstraction for experiments.

##### Available experiments
- [general training experiment](sapsan/lib/experiments/training.py)
- [evaluation 1d experiment](sapsan/lib/experiments/evaluation_flatten.py)
- [evaluation 3d encoder experiment](sapsan/lib/experiments/evaluation_3d.py)
- [evaluation 3d autoencoder experiment](sapsan/lib/experiments/evaluation_autoencoder.py)

##### How to implement new experiment:

Extend `Experiment` class and impement `run` method.

```python
from sapsan.core.models import Experiment, ExperimentBackend


class AwesomeExperiment(Experiment):
    def __init__(self, name: str, backend: ExperimentBackend):
        super().__init__(name, backend)
        
    def run(self):
        # do whatever you need to execute during experiment
        return {}
```

#### Tracking backend

General abstraction for experiment tracker.

##### Available tracking backends
- [MlFlow](sapsan/lib/backends/mlflow.py)
- [FakeBackend](sapsan/lib/backends/fake.py)

##### How to implement new experiment:

Extend `ExperimentBackend` class and impement `log_metric`, `log_parameter`, `log_artifact`  methods.

```python
from sapsan.core.models import ExperimentBackend


class InMemoryTrackingBackend(ExperimentBackend):
    def __init__(self, name: str):
        super().__init__(name)
        self.metrics = []
        self.parameters = []
        self.artifacts = []

    def log_metric(self, name, value):
        self.metrics.append((name, value))

    def log_parameter(self, name, value):
        self.metrics.append((name, value))

    def log_argifact(self, path):
        self.artifacts.append(path)
```


-------
### Examples

Examples of implemented experiments.

- [3d audoencoder example](./examples/autoencoder_example.py)
- [3d convolution encoder example](./examples/cnn_example.py)
- [KRR 1d estimator](./examples/krr_example.py)

-------

### Kubeflow

[Docs for experiments in kubeflow](./docs/kubeflow.md)

-------

© (or copyright) 2019. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.
