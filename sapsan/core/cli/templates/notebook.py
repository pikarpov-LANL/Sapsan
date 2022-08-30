TEMPLATE = '''#!/usr/bin/env python
# coding: utf-8
# In[ ]:
import os
import sys

from sapsan.lib.backends import MLflowBackend
from sapsan.lib.data import HDF5Dataset, EquidistantSampling, flatten
from sapsan.lib import Train, Evaluate

#Importing your custom estimator, i.e. ML algorithm
from {name}_estimator import {name_upper}, {name_upper}Config

# In[ ]:
#--- Experiment tracking backend ---

#MLflow - the server will be launched automatically
#in case it won't, type in cmd: mlflow ui --port=9000
#uncomment tracking_backend to use mlflow

experiment_name = "{name} experiment"
#tracking_backend = MLflowBackend(experiment_name, host="localhost", port=9000)

# In[ ]:
#--- Data setup ---
#In the intereset of loading and training multiple timesteps
#one can specify which checkpoints to use and where
#they appear in the path via syntax: {{checkpoint:format}}
#
#Next, you need to specify which features to load; let's assume 
#        path = "{{feature}}.h5"
#
# 1) If in different files, then specify features directly;
#    The default HDF5 label will be the last label in the file
#    Ex: features = ['velocity', 'denisty', 'pressure']
# 2) If in the same, then duplicate the name in features
#    and specify which labels to pull
#    Ex: features = ["data", "data", "data"]
#        feature_labels = ['velocity', 'density', 'pressure']
#
#Example: path = "data/t{{checkpoint:1.0f}}/{{feature}}_dim32_fm15.h5"

path = "./data/"

#a list of training features
features = []

#a lits of target features
target = []

#Dimensionality of your data in format (D,H,W)
INPUT_SIZE = 

#Reduce dimensionality to the following in format (D,H,W)
SAMPLE_TO = 

#Sampler to use for reduction
sampler = EquidistantSampling(SAMPLE_TO)

# In[ ]:
#Load the data
#>>>Specify which checkpoints to load!<<<
data_loader = HDF5Dataset(path=path,
                          features=features,
                          target=target,
                          checkpoints=[],
                          input_size=INPUT_SIZE,
                          sampler=sampler)

x, y = data_loader.load_numpy()
y = flatten(y)

#convert_to_torch takes in a list or a numpy array
loaders = data_loader.convert_to_torch([x, y])

# In[ ]:
#Machine Learning model to use

#Configuration of the model parameters:
#    n_epochs = number of epochs (iterations)
#    patience = number of epochs to run beyond convergence
#    min_delta = loss based convergence cut-off
estimator = {name_upper}(
    config={name_upper}Config(n_epochs=5, patience=10, min_delta=1e-5)
)

# In[ ]:
#--- Train the model ---
#Set the experiment
training_experiment = Train(model=estimator,
                            #backend=tracking_backend, #uncomment to use mlflow
                            loaders = loaders,
                            data_parameters = data_loader)

#Train the model
model = training_experiment.run()

# In[ ]:
get_ipython().run_line_magic('matplotlib', 'inline')

#--- Test the model ---
#Load the test data
#>>>Specify which checkpoints to load!<<<
data_loader = HDF5Dataset(path=path,
                          features=features,
                          target=target,
                          checkpoints=[],
                          input_size=INPUT_SIZE,
                          sampler=sampler)
x, y = data_loader.load_numpy()

#Set the test experiment
evaluation_experiment = Evaluate(model = model,
                                 #backend=tracking_backend, #uncomment to use mlflow                                 
                                 loaders = [x, y],
                                 data_parameters = data_loader)


#Test the model
target_cube, pred_cube = evaluation_experiment.run()
'''

def get_template(name: str):
    return TEMPLATE.format(name=name.lower(),
                           name_upper=name.capitalize())