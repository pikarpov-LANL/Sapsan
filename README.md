# Sapsan



### Intro

TODO


### Kubeflow

Docs for experiments in [kubeflow](./docs/kubeflow.md)

### POC example

TODO

```python
from sapsan.general.experiment import MlflowExperiment
from sapsan.general.data.dataset import JHTDBDataset
from sapsan.general.estimator.krr import KrrEstimatorConfiguration, KrrEstimator
from sapsan.utils.plot import PlotUtils

# create dataset
dataset = JHTDBDataset('somepath')
# create model
estimator = KrrEstimator(KrrEstimatorConfiguration(0.001, 1.789))
#create experiment
experiment = MlflowExperiment(estimator=estimator, dataset=dataset, callbacks=[
    PlotUtils.plot_histograms,
    PlotUtils.plot_slices
])
# run experiment
experiment.run()
# get report on experiment run
experiment.get_report()
```


### TODOs

- [ ] implement refactored dataset loaders and estimators
- [ ] tests
- [ ] CI using Github actions
- [ ] version relase
- [ ] docker images
- [ ] pipeline to kubeflow
- [ ] documentation



© (or copyright) 2019. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.
