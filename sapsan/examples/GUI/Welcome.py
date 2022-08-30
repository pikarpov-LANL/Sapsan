import streamlit as st

st.set_page_config(
    page_title="Welcome",
    page_icon="🚅",
)

st.sidebar.success("Select an experiment above")

st.markdown(

    """
    # Welcome to Sapsan!

    ---

    Sapsan is a pipeline for Machine Learning (ML) based turbulence modeling. While turbulence 
    is important in a wide range of mediums, the pipeline primarily focuses on astrophysical application. 
    With Sapsan, one can create their own custom models or use either conventional or physics-informed 
    ML approaches for turbulence modeling included with the pipeline ([estimators](https://sapsan-wiki.github.io/details/estimators/)).
    For example, Sapsan features ML models in its set of tools to accurately capture the turbulent nature applicable to Core-Collapse Supernovae.

    > ## **Purpose**

    > Sapsan takes out all the hard work from data preparation and analysis in turbulence 
    > and astrophysical applications, leaving you focused on ML model design, layer by layer.             

    **👈 Select an experiment from the dropdown on the left** to see what Sapsan can do!
    ### Want to learn more?
    - Check out Sapsan on [Github](https://github.com/pikarpov-LANL/Sapsan)
    - Find the details on the [Wiki](https://sapsan-wiki.github.io/)
"""
)

with st.expander('License Information'):
    st.markdown(
        """
Sapsan has a BSD-style license, as found in the [LICENSE](https://github.com/pikarpov-LANL/Sapsan/blob/master/LICENSE) file.            

© (or copyright) 2019. Triad National Security, LLC. All rights reserved. This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
    """
    )