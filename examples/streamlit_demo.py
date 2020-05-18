import streamlit as st

import os
import sys
import inspect

sys.path.append("../")

from sapsan.lib.backends.fake import FakeBackend
from sapsan.lib.backends.mlflow import MLflowBackend
from sapsan.lib.data.jhtdb_dataset import JHTDB128Dataset
from sapsan.lib.data import Equidistance3dSampling
from sapsan.lib.estimator import CNN3d, CNN3dConfig
from sapsan.lib.estimator.cnn.spacial_3d_encoder import CNN3dModel
from sapsan.lib.experiments.evaluate_3d import Evaluate3d
from sapsan.lib.experiments.train import Train

import pandas as pd
import hiddenlayer as hl
import torch
import matplotlib.pyplot as plt


experiment_name = st.sidebar.text_input("Experiment name", "Awesome experiment")
dataset_name = st.sidebar.selectbox("Dataset", ["JHTDB128Dataset"])
features = st.sidebar.multiselect("Features", ["u", "b"], "u")
target = st.sidebar.multiselect("Labels", ["u"], "u")
checkpoint_size = st.sidebar.selectbox("Checkpoint data size", [32, 16, 8], 0)
sample_to = st.sidebar.selectbox("Sample to", [16])
grid_size = st.sidebar.selectbox("Grid size", [8, 16, 32], 0)
checkpoints = st.sidebar.multiselect("Checkpoints", [0], 0)
n_epochs = st.sidebar.slider("Number of epochs", 1, 100, 1)

st.write("Run parameters")
st.table(pd.DataFrame([
    ["dataset", dataset_name],
    ["number of epochs", n_epochs],
    ["checkpoints size", checkpoint_size],
    ["sample to", sample_to],
    ["grid size", grid_size],
    ["features", features],
    ["labels", target],
    ["checkpoints", checkpoints]
], columns=["key", "value"]).T)

estimator = CNN3d(
    config=CNN3dConfig(n_epochs=n_epochs, grid_dim=grid_size)
)


def run_experiment():
    tracking_backend = FakeBackend(experiment_name)
    path = "data/t{checkpoint:1.0f}/{feature}_dim32_fm15.h5"
    sampler = Equidistance3dSampling(checkpoint_size, sample_to)

    st.write("Estimator created...")
    x, y = JHTDB128Dataset(path=path,
                           features=features,
                           target=target,
                           checkpoints=[0],
                           grid_size=grid_size,
                           checkpoint_data_size=checkpoint_size,
                           sampler=sampler).load()
    st.write("Dataset loaded...")
    training_experiment = Train(name=experiment_name,
                                backend=tracking_backend,
                                model=estimator,
                                inputs=x, targets=y)
    st.write("Runnning experiment...")
    training_experiment.run()
    st.write("Done!")


if st.checkbox("Show code of model"):
    st.code(inspect.getsource(CNN3dModel))

if st.checkbox("Show model graph"):
    res = hl.build_graph(estimator.model, torch.zeros([72, 1, 2, 2, 2]))
    st.graphviz_chart(res.build_dot())


if st.button("Run experiment"):
    st.write("Experiment is running. Please hold on...")
    run_experiment()



