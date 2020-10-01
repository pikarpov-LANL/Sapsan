import inspect
import textwrap
from collections import OrderedDict

import streamlit as st
import st_experiments as experiments
import st_custom as custom

EXPERIMENTS = OrderedDict(
    [
        ("Custom", (custom.custom, None)),
        ("Examples", (experiments.cnn3d, None)),
        ("Welcome", (experiments.intro, None)),
        ("test", (experiments.test, None)),
        ("1D CCSN", (experiments.ccsn, None)),
        
    ]
)

def run():
    experiment_name = st.sidebar.selectbox("Choose an experiment", list(EXPERIMENTS.keys()),0)
    experiment = EXPERIMENTS[experiment_name][0]
    
    if experiment_name == 'Welcome':
        show_code = False
        st.markdown("# Welcome to Sapsan!")
    else:
        pass
    
    experiment()


if __name__ == "__main__":
    run()