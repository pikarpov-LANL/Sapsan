import inspect
import textwrap
from collections import OrderedDict

import streamlit as st
from pages.st_welcome import welcome
from pages.st_cnn3d import cnn3d

EXPERIMENTS = OrderedDict(
    [
        ("Welcome", (welcome, None)),
        ("Examples", (cnn3d, None)),
    ]
)

def run():
    experiment_name = st.sidebar.selectbox("Choose an experiment", list(EXPERIMENTS.keys()),0)
    experiment = EXPERIMENTS[experiment_name][0]
    
    if experiment_name == 'Welcome':
        show_code = False
    else:
        pass
    
    experiment()


if __name__ == "__main__":
    run()