import streamlit as st
import os
import sys
import inspect

#uncomment if cloned from github!
sys.path.append("/home/pkarpov/Sapsan/")

from sapsan.lib.backends.fake import FakeBackend
from sapsan.lib.backends.mlflow import MLflowBackend
from sapsan.lib.data.hdf5_dataset import HDF5Dataset
from sapsan.lib.data import EquidistanceSampling
from sapsan.lib.estimator import CNN3d, CNN3dConfig
from sapsan.lib.estimator.cnn.spacial_3d_encoder import CNN3dModel
from sapsan.lib.experiments.evaluate_3d import Evaluate3d
from sapsan.lib.experiments.train import Train

import pandas as pd
import hiddenlayer as hl
import torch
import matplotlib.pyplot as plt

st.title('Sapsan Configuration')
st.write('Intro text lipsum')

#--- Experiment tracking backend ---

add_selectbox = st.sidebar.markdown(
 "General Configuration"  
)

#--- Default Parameters ---
experiment_name = "CNN experiment"
MLFLOW_HOST = 'localhost'
MLFLOW_PORT = '9000'
AXIS = 3
CHECKPOINT_DATA_SIZE = 32
SAMPLE_TO = 16
GRID_SIZE = 8
n_epochs=1
patience=10
min_delta=1e-5
path = "data/t{checkpoint:1.0f}/{feature}_dim32_fm15.h5"
features = 'u'
target = 'u'
tracking_backend = FakeBackend(experiment_name)
sampler = EquidistanceSampling(CHECKPOINT_DATA_SIZE, SAMPLE_TO, AXIS)
estimator = CNN3d(config=CNN3dConfig(n_epochs=n_epochs, grid_dim=GRID_SIZE, 
                                     patience=patience, min_delta=min_delta))
#--- End Default ---

experiment_name = st.sidebar.text_input('experiment name', "CNN experiment", type='default')

show_backend_config = st.sidebar.checkbox('Backend', value=False)
if show_backend_config:
    backend_selection = st.sidebar.selectbox(
        'What backend to use?',
        ('Fake', 'MLflow')
    )

    if backend_selection == 'Fake':
        tracking_backend = FakeBackend(experiment_name)

    elif backend_selection == 'MLflow':
        MLFLOW_HOST = st.sidebar.text_input("mlflow host", "localhost", type='default')
        MLFLOW_PORT = st.sidebar.number_input("mlflow port", 1024,65535,9999)
        tracking_backend = MLflowBackend(experiment_name, MLFLOW_HOST, MLFLOW_PORT)
    

show_data_config = st.sidebar.checkbox('Data', value=False)

if show_data_config:
    path = st.sidebar.text_input("data path", "data/t{checkpoint:1.0f}/{feature}_dim32_fm15.h5", type='default')
    features = st.sidebar.text_input("features", 'u').split(",")
    target = st.sidebar.text_input("target", 'u')

    #Dimensionality of your data per axis
    CHECKPOINT_DATA_SIZE = st.sidebar.number_input("Data size per dimension", min_value=1, value=32)

    #Reduce dimensionality of each axis to
    SAMPLE_TO = st.sidebar.number_input("Reduce each dimension to", min_value=1, value=16)

    #Dimensionality of each axis in a batch
    GRID_SIZE = st.sidebar.number_input("Batch size per dimension", min_value=1, value=8)

    #Sampler to use for reduction
    sampler_selection = st.sidebar.selectbox('What sampler to use?', ('Equidistant3D', ''))
    if sampler_selection == "Equidistant3D": sampler = EquidistanceSampling(CHECKPOINT_DATA_SIZE, 
                                                                            SAMPLE_TO, AXIS)

        
show_model_config = st.sidebar.checkbox('Model', value=False)

if show_model_config:
    #--- Train the model ---
    #Machine Learning model to use
    
    n_epochs = st.sidebar.number_input("# epochs", min_value=1, value=10)
    grid_dim = GRID_SIZE
    patience = st.sidebar.number_input("patience", min_value=0, value=10)
    min_delta = float(st.sidebar.text_input("min_delta", "1e-5"))
    
    estimator = CNN3d(config=CNN3dConfig(n_epochs=n_epochs, grid_dim=GRID_SIZE, 
                                         patience=patience, min_delta=min_delta))


st.write("Run parameters")
st.table(pd.DataFrame([
    #"dataset", dataset_name],
    ["number of epochs", n_epochs],
    #["checkpoints size", checkpoint_size],
    ["sample to", SAMPLE_TO],
    ["grid size", GRID_SIZE],
    ["features", features],
    ["target", target],
    #["checkpoints", checkpoints]
], columns=["key", "value"]).T)


def run_experiment():
    #Load the data
    x, y = HDF5Dataset(path=path,
                       features=features,
                       target=target,
                       checkpoints=[0],
                       grid_size=GRID_SIZE,
                       checkpoint_data_size=CHECKPOINT_DATA_SIZE,
                       sampler=sampler).load()
    st.write("Dataset loaded...")
    
    #Set the experiment
    training_experiment = Train(name=experiment_name,
                                 backend=tracking_backend,
                                 model=estimator,
                                 inputs=x, targets=y)
    #Train the model
    st.write("Runnning experiment...")
    training_experiment.run()
    
    st.write('Done!')

#def evaluate_experiment():
    #--- Test the model ---
    #Load the test data
    x, y = HDF5Dataset(path=path,
                       features=features,
                       target=target,
                       checkpoints=[0],
                       grid_size=GRID_SIZE,
                       checkpoint_data_size=CHECKPOINT_DATA_SIZE,
                       sampler=sampler).load()

    #Set the test experiment
    evaluation_experiment = Evaluate3d(name=experiment_name,
                                               backend=tracking_backend,
                                               model=training_experiment.model,
                                               inputs=x, targets=y,
                                               grid_size=GRID_SIZE,
                                               checkpoint_data_size=SAMPLE_TO)

    #Test the model
    evaluation_experiment.run()
    

    data = y
    #'data', data
    st.pyplot()
    
    
if st.checkbox("Show model graph"):
    res = hl.build_graph(estimator.model, torch.zeros([72, 1, 2, 2, 2]))
    st.graphviz_chart(res.build_dot())
    
if st.checkbox("Show code of model"):
    st.code(inspect.getsource(CNN3dModel))
    
if st.button("Run experiment"):
    #st.write("Experiment is running. Please hold on...")
    run_experiment()
    
#if st.button("Evaluate experiment"):
#    #st.write("Experiment is running. Please hold on...")
#    evaluate_experiment()
    
    
   



