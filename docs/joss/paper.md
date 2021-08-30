---
title: 'Sapsan: Framework for Supernovae Turbulence Modeling with Machine Learning'
tags:
  - Python
  - machine learning
  - astronomy
  - supernovae
  - turbulence
authors:
  - name: Platon I. Karpov
    orcid: 0000-0003-4311-8490
    affiliation: 1, 2
  - name: Iskandar Sitdikov
    orcid: 0000-0002-6809-8943
    affiliation: 3
  - name: Chengkun Huang
    orcid: 0000-0002-3176-8042
    affiliation: 2
  - name: Chris L. Fryer
    orcid: 0000-0003-2624-0056
    affiliation: 2
affiliations:
  - name: Department of Astornomy & Astrophysics, University of California, Santa Cruz, CA
    index: 1
  - name: Los Alamos National Laboratory, Los Alamos, NM
    index: 2
  - name: Provectus IT Inc., Palo Alto, CA
    index: 3
date: 4/8/2021
bibliography: paper.bib
---

# Summary
[Sapsan](https://github.com/pikarpov-LANL/Sapsan) is a framework to make Machine Learning (ML) more accessible in the study of turbulence, focusing on astrophysical applications. Sapsan includes modules to load, filter, subsample, batch, and split the data from hydrodynamic (HD) simulations for training and validation. Next, the framework includes built-in conventional and physics-informed published estimators that were used for turbulence modeling. This ties into Sapsan's custom estimator module, aimed at designing a custom ML model layer-by-layer, which is the core benefit of using the framework. To share your custom model, every new project created via Sapsan comes with pre-filled, ready-for-release Docker files. Furthermore, training and evaluation modules come with Sapsan as well. The latter, among other features, includes the construction of power spectra and comparison to established analytical turbulence closure models, such as a gradient model. Thus, Sapsan takes out all the hard work from data preparation and analysis, leaving one focused on the ML model design itself.

# Statement of Need

It has been challenging for domain sciences to adopt Machine Learning (ML) for their respective projects, particularly for physical simulations modeling turbulence. It is rather challenging to prove that an ML model has arbitrarily learned the laws of physics embedded into the problem with the ability to extrapolate within the parameter-space of the simulation. The inability to directly infer the prediction capabilities of ML is one of the major causes behind the slow adaption rates; however, the community cannot ignore the effectiveness of ML.

Turbulence is ubiquitous in astrophysical environments; however, it involves physics at a vast range of temporal and spatial scales, making accurate fully-resolved modeling difficult. Various analytical turbulence models have been developed to be used in simulations using time or spatial averaged governing equations, such as in RANS (Reynolds-averaged Navier-Stokes) and LES (Large Eddy Simulation), 
but accuracy is lacking. In search of better methods to model turbulence in core-collapse supernovae, it became apparent that ML has great potential to produce more accurate turbulence models on an unaveraged subgrid-scale than the current methods. Scientists from both industry and academia [@king2016], [@zhang2018] have already begun using ML for applied turbulent problems. Still, none reach out to a theoretical medium of physics and astronomy community on a practical level. For example, physics-based model evaluation and interpretability tools are not standardized, nor are they widely available. As a result, it is a common struggle to verify published results, with the setup not fully disclosed, the poorly structured code lacking clear commenting, or even worse - not publicly available; the problem ML community can relate to as well [@Hutson725]. Thus, it is not surprising that there is considerable skepticism against ML in physical sciences, with astrophysics being no exception [@carleo2019].

In pursuit of our supernova (SNe) study, the issues outlined above became painfully apparent. Thus, we attempted to simplify the barrier to entry for new researchers in domain science fields studying turbulence to employ ML, with the main focus on astrophysical applications. As a result, an ML python-based pipeline called ``Sapsan`` has been developed. The goals have 
been to make it accessible and catered to the community through Jupyter Notebooks, command-line-interface (CLI) and graphical-user-interface 
(GUI)\footnote{demo available at [sapsan.app](https://sapsan.app/])} 
available for the end-user. ``Sapsan`` includes built-in optimized ML models for turbulence treatment, both conventional and physics-based. More importantly, at its core, the framework is meant to be flexible and modular; hence there is an intuitive interface for users to work on their own ML algorithms. Most of the mundane turbulence ML researcher needs, such as data preprocessing and prediction analysis, can be automated through Sapsan, with a streamlined process of custom estimator development. In addition, ``Sapsan`` brings best practices from the industry regarding ML development frameworks. For example, ``Sapsan`` includes docker containers for reproducible release, as well as 
[MLflow](https://mlflow.org/) for experiment tracking. Thus, ``Sapsan`` is a single, complete interface for turbulence ML-based research.

`Sapsan` is distributed through [Github](https://github.com/pikarpov-LANL/Sapsan) and [pip](https://pypi.org/project/sapsan/). For further reference, [wiki](https://github.com/pikarpov-LANL/Sapsan/wiki) is maintained on Github as well.

# Framework
``Sapsan`` organizes workflow via three respective stages: data preparation, machine learning, and analysis. The whole process can be further wrapped in Docker to publish your results for reproducibility, as reflected by Figure 1. Let's break down each stage in the context of turbulence subgrid modeling, e.g., a model to predict turbulent behavior at the under-resolved simulation scales.

* __Data Preparation__
  * __Loading Data:__ ``Sapsan`` is ready to process common 2D & 3D hydrodynamic (HD) and magnetohydrodynamic (MHD) turbulence data in simulation-code-specific data formats, such as HDF5 (with more to come per community need).
  * __Transformations:__ a variety of tools are ready for you to ready the data for training
     * __Filter:__ to build a subgrid model, one will have to filter the data, e.g., remove small-scale perturbations. 
    A few examples would be either a box, spectral, or a gaussian filter.  The data can be filtered on the fly within the framework.
     * __Sample:__ to run quick tests of your model, you might want to test on a sampled version of the data while retaining the full spacial domain. Thus, equidistant sampling is ready to go in ``Sapsan``.
     * __Batch:__ the pipeline will spatially batch the data
     * __Split:__ data is divided into testing and validation datasets
    
* __Machine Learning__

  * __Model Setup:__ different ML models would be appropriate for different physical regimes. ``Sapsan`` provides templates for a selection of both conventional and physics-based models with more to come. Only important options are left up to the user to edit, with most overhead kept in the backend.
     * __Layers:__  define and order the ML layers
     * __Tracking:__ add extra parameters to be tracked by MLflow
     * __Loss Function:__ decide on a conventional or physics-based/custom loss function to model the data
     * __Optimizer:__ choose how to optimize the training
     * __Scheduler:__ select how to adjust the learning rate

* __Analysis__
  *  __Trained Model:__ a turbulence subgrid model telling us how small-scale structure affects the large scale quantities, i.e., it completes or ''closes'' the governing large-scale equations of motion with small-scale terms. The prediction from a trained ML model is used to provide the needed quantities.
  *  __Analytical Tools:__ compare the trained model with conventional analytic turbulence models, such as Dynamic Smagorisnky [@lilly1966] or Gradient model [@]. Furthermore, conduct other physics-based tests, for example, compute the power spectrum of your prediction.

For further information on each stage, please refer to [Sapsan's Wiki on Gihub](https://github.com/pikarpov-LANL/Sapsan/wiki).

![High-level overview of ``Sapsan's`` workflow.](Sapsan_highlevel_overview.png)

### Dependencies
Here only the core dependencies will be covered. Please refer to [GitHub](https://github.com/pikarpov-LANL/Sapsan) for the complete list.

* __Training__ 
   * __PyTorch:__ `Sapsan`, at large, relies on PyTorch to configure and train ML models. Thus, the parameters in the aforementioned __Model Set Up__ stage should be configured with PyTorch functions. [Convolutional Neural Network (CNN)](https://github.com/pikarpov-LANL/Sapsan/blob/master/sapsan/examples/cnn_example.ipynb) and [Physics-Informed Convolutional Auto Encoder (PICAE)](https://github.com/pikarpov-LANL/Sapsan/blob/master/sapsan/examples/picae_example.ipynb) examples included with `Sapsan` are based on PyTorch. [@pytorch]
      * __Catalyst:__ used as part of the backend to configure early-stopping of the model and logging [@catalyst]
   * __Scikit-learn:__ the framework supports it, as shown in the [Kernel Ridge Regression (KRR)](https://github.com/pikarpov-LANL/Sapsan/blob/master/sapsan/examples/krr_example.ipynb) example in `Sapsan`. However, due to lack of scalability and features, it is advised to use PyTorch based setup. [@scikit-learn]

* __Tracking__
   * __MLflow:__ allows for tracking across large quantities of training and evaluation runs through an intuitive web interface. Beyond the few default parameters, a user can include custom parameters to be tracked. [@mlflow_github]

* __Interface__
    * __Jupyter Notebook:__ the most direct and versatile way to use `Sapsan`
    * __Streamlit (GUI):__ a graphical user interface (GUI) for `Sapsan`. While lacking in its flexibility, it is perfect for constructing a demo to present your project. To try it out, please visit [sapsan.app](https://sapsan.app) [@streamlit2019]
    * __Click (CLI):__ a command-line interface (CLI) for `Sapsan`. It is used to get the user up and running with templates for a custom project. [@click]



# Applications

While ``Sapsan`` is built to be highly customizable for a wide variety of projects in physical sciences, it is optimized for the study of turbulence. 

## Hydro simulations

Here are a few examples of a turbulence closure model trained on the high-resolution Johns Hopkins Turbulence Database (JHTDB) [@jhtdb2008]. The dataset used in this comparison is a direct numerical simulation (DNS) of a statistically-stationary isotropic 3D MHD
turbulence dataset, 1024^3 in spatial resolution and covering roughly one large eddy turnover time over 1024 checkpoints, e.i. dynamical time of the system
[@Eyink2013]. We compare it with a commonly used Dynamic Smagorinsky (DS) turbulence closure model [@lilly1966]. On ``Sapsan``  side, a Kernel
Ridge Regression model [@murphy2004] is used to demonstrate the effectiveness of conventional ML approaches in tackling turbulence
problems. In this test, we used the following setup:


* __Train features:__ velocity (*u*), vector potential (*A*), magnetic field (*B*), and their respective derivatives at timestep = 1. All quantities have been filtered down to 15 fourier modes to remove small-scale perturbations, mimicking the lower fidelity of a non-DNS simulation.
* __Model Input:__ low fidelity velocity (*u*), vector potential (*A*), magnetic field (*B*), and their respective derivatives at a set timestep in the future.
* __Model Output:__ velocity stress tensor ($\tau$) at the matching timestep in the future, which effectively represents the difference between large and small scale structures of the system.

In Figure 2, it can be seen that the ML-based approach significantly outperforms the DS subgrid model in reproducing the probability density function, i.e., a statistical distribution of the stress tensor. The results are consistent with [@king2016].

![Predicting turbulent stress-tensor in statistically-stationary isotropic MHD turbulence setup. The outmost left plot compares the original spatial map of the stress-tensor component to the plot in the middle depicting the predicted spatial map. The plot on the right presents probability density functions (PDF), i.e., distributions, of the original stress-tensor component values, the ML predicted values, and the conventional DS subgrid model prediction.](JHTDB.png)

## Supernovae
Even though the conventional regression-based ML approach worked well in the previous section, the complexity of our physical problem forced us to seek a more rigorous ML method. Supernovae host a
different physical regime that is far from the idealistic MHD turbulence case from before. Here we are dealing with dynamically
changing statistics and evolution of the turbulence that is not necessarily isotropic. Turbulence can behave drastically differently depending on the evolutionary stage; hence, a more sophisticated model is required. With ``Sapsan``, we have tested a 3D CNN (Convolutional Neural Network) model in an attempt to predict a turbulent velocity stress tensor in a realistic Core-Collapse Supernova (CCSN) case. Figure 3 presents results of the following:

* __Train features:__ velocity (*u*), magnetic field (*B*), and their respective derivatives at timestep = 500 (halfway of the total simulation). All quantities have been filtered down to 15 fourier modes to remove small-scale perturbations, mimicking the lower fidelity of a non-DNS simulation.
* __Model Input:__ low fidelity velocity (*u*), magnetic field (*B*), and their respective derivatives at a set timestep in the future.
* __Model Output:__ velocity stress tensor ($\tau$) at the matching timestep in the future, which effectively represents the difference between large and small scale structures of the system.

In this case, the matching level of the distributions is the most critical factor. It can be seen that the probability density functions match quite closely, with the outlier at the most negative values being the exception, even though the prediction is performed far into the future (timestep=1000, end of the simulation time). For further discussion on the comparison of the models and the results, please refer to [the ApJ paper].

![Predicting Turbulent Stress-Tensor in a Core-Collapse Supernovae (CCSN). The model has been trained on a 3D MHD direct numerical simulation (DNS) of the first 10ms after the shockwave bounced off the core in a CCSN scenario. On the left, the two figures are the 2D slices of a 3D data-cube prediction, with the right plot presenting a comparison of PDFs of the original 3D data, 3D ML prediction, and a conventional Gradient subgrid model.](ccsn_mri_plots.png)

# Acknowledgements
The development of ``Sapsan`` was supported by the Laboratory Directed Research and Development program and the Center for Space and Earth Science at Los Alamos National Laboratory through the student fellow grant. In addition, We would like to thank DOE SciDAC for additional funding support.

# References
