# Sapsan  <a href="http://sapsan.app"><img src="https://github.com/pikarpov-LANL/Sapsan/blob/master/docs/images/logo3_black_slim_notitle.png?raw=true"  alt="Sapsan logo" align="right" width="100"></a>

* [Intro](#intro) 
* [Structure](#structure)
  * [Estimator](#estimator)
  * [Experiment](#experiment)
  * [Dataset](#dataset)
  * [Tracking backend](#tracking-backend) 
* [Examples](#examples)
* [CLI](#cli)
* [Kubeflow](#kubeflow)
-------

### Intro
Sapsan is a pipeline for easy Machine Learning implementation in scientific projects. That being said, its primary goal and featured models are geared towards dynamic MHD turbulence subgrid modeling. Sapsan will soon feature Physics-Informed Machine Learning models in its set of tools to accurately capture the turbulent nature applicable to Core-Collapse Supernovae.

Feel free to check out a website version at [sapsan.app](http://sapsan.app). The interface is indentical to the GUI of the local version of Sapsan, except lacking the ability to edit the model code on the fly.

Note: currently Sapsan is in alpha, but we are actively working on it and introduce new feature on a daily basis.

### Getting Started

To get started, clone this repo and install the requirements

```shell script
git clone https://github.com/pikarpov-LANL/Sapsan.git
cd Sapsan/
pip install -r requirements.txt
```

If you want to use the GPU enabled version then change the last line to
```shell script
pip install -r requirements_gpu.txt
```

or you can install sapsan via pip
```shell script
pip install sapsan
```
##### Graphical Interface
We've built a Sapsan configuration and running interface with Streamlit. In order to run it type in the following and follow the instrucitons - the interface will be opened in your browser.
```shell script
streamlit run examples/GUI/st_intro.py
```

##### Command Line Interface
Please run an example to make sure everything has been installed correctly. It is a jupyter notebook which can be found here:
```shell script
Sapsan/examples/cnn_example.ipynb
```
In order to get started on your own project, you can use the command-line-interface interface:
```shell script
sapsan create --name awesome
```

-------

### Structure
Structure of project is build around few concepts making this project easier to extend to more cases.
Core abstractions are: estimator, dataset, experiment, tracking backend
Core abstraction are defined in [models.py](sapsan/core/models.py) file.

#### Estimator
General abstraction for models/algorithms.

##### Available estimators
- [KRR 1d estimator](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html)
- [3d convolution encoder estimator](https://pytorch.org/docs/stable/nn.html#conv3d)
- [3d autoencoder estimator](sapsan/lib/estimator/cnn/spacial_autoencoder.py) *(coming soon!)*

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
- [2d dataset](sapsan/lib/data/flatten_dataset.py)

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

- [3d convolution encoder example](./examples/cnn_example.ipynb)
- [3d audoencoder example](./examples/autoencoder_example.py) *(coming soon!)*
- [KRR 1d estimator](./examples/krr_example.py) *(coming soon!)*

### CLI

To use structure of Sapsan and CI/CD capabilities run
```shell script
sapsan create <NAME>
cd <NAME>
git init
git remote add origin <YOUR_REPOSITORY_ORIGIN>
git add .
git commit -m "Initial commit"
git push origin master
```

`<NAME>` can be `research` for example.


-------

#### Examples

##### Local via docker compose

```shell script
docker-compose build
docker-compose up --force-recreate
```

Then open browser at [localhost:8888](http://localhost:8888)


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
