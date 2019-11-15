# Sapsan


 - name='test'
>> Name of the run, as it will show up in the scores.dat and stats.dat
 - parameters=None
>> parameters (e.g. features) to train on. The last one in the array will be the target feature.
 - path='/raid1/JHTDB'
>> path to the data
 - dataset='iso'
>> which data set to use; options are 'iso' for HD turbulence, and 'mhd' for MHD turbulence
 - savepath='Figures'
>> where to save the data
 - test=False
>> set to True if want to randomly split the same data into training and testing set.
 - alpha=None
>> will use the default KRR alpha
 - gamma=None
>> will use the default KRR gamma
 - dim=128
>> dimension of the training set
 - max_dim=512
>> initial dimension of the data file, from which training and testing sets will be extracted
 - step=None
>> spatial separation between each data point used for training
 - t=[0]
>> training timestep
 - ttest=[0]
>> testing timestep
 - dt=2.5e-3
>> numeral value for dt to calculate the actual time of each timestep
 - fm=15
>> number of modes to filter down to
 - filtname='spectral'
>> filter to use; available options also include 'boxfilt'
 - tnc=2
>> which tensor component to predict; can be set to 0, 1, or 2
 - axis=2
>> dimensionality; either 2 or 3
 - from3D=False
>> extracts 2D slice from 3D data; only relevant if axis=2
 - outliers=False
>> only include outliers when adding consecutive steps, thus the data from additional timesteps won't be sampled in the same way as the 1st timestep.



© (or copyright) 2019. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.
