# Sapsan

| Parameter                             | Description                                | Default                                         |
| ------------------------------------- | ------------------------------------------ | ----------------------------------------------- |
| `name` | name of the run, as it will show up in the scores.dat and stats.dat | 'test' |
| `parameters` | parameters (e.g. features) to train on | None |
| `target` | target parameter to train against and to predict | None |
| `path` | path to the data | '/raid1/JHTDB' |
| `dataset` | which data set to use; options are 'iso' for HD turbulence, and 'mhd' for MHD turbulence | 'iso' |
| `savepath` | where to save the data | 'Figures' |
| `alpha` | will use the default KRR alpha | None |
| `gamma` |  will use the default KRR gamma | None |
| `dim` |  dimension of the training set | 128 |
| `max_dim` |  initial dimension of the data file, from which training and testing sets will be extracted | 512 |
| `step` | spatial separation between each data point used for training; default is to choose the step to cover the whole domain with equidistent point separation | None |
| `ttrain` | training timestep | [0] |
| `dt` |  numeral value for dt to calculate the actual time of each timestep | 1 |
| `fm` | number of modes to filter down to | 15 |
| `filtname` |  filter to use; available options also include 'boxfilt' | 'spectral' |
| `axis` |  dimensionality; either 2 or 3 | 2 |
| `from3D` |  extracts 2D slice from 3D data; only relevant if axis=2 | False |
| `outliers` |  only include outliers when adding consecutive steps, thus the data from additional timesteps won't be sampled in the same way as the 1st timestep | False |
| `datatype` | what loader to choose; options also include .txt, .dat | 'h5' |
| `parLabel` | label of the training dataset in the hdf5 container; default is to use [-1] dataset | None | 
| `targetLabel` | label of the target dataset in the hdf5 container; default is to use [-1] dataset | None | 
| `n_epochs` | the number of epochs to iterate over | 100 |
| `batch_size` | the number of training batches | 12 |
| `cube_size` | the size of the cubes to split the data into | 8 |
| `method` | method to use for training; either 'cnn' or 'krr' | 'krr' |
| `experiment_name` | name of the experiment for mlflow tracking | 'demo' |


© (or copyright) 2019. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.
