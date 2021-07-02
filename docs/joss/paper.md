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
[Sapsan](https://github.com/pikarpov-LANL/Sapsan) is a framework to make Machine Learning (ML) more accessible in the study of turbulence, with a focus on astrophysical applications. Sapsan includes modules to load, filter, sample down, batch, and split your data for training and validation. Furthermore, the evoluation modules, among other features, include construction of power spectra and comparison to established analytical turbulence closure models, such as a gradient model. Thus, Sapsan takes out all the hard work from data preparation and analysis, leaving one focused on ML model design, layer by layer.

# Statement of Need

It has been challenging for domain sciences to adapt Machine Learning (ML) for their respective projects. 
From the authors' background, it has been particularly difficult to employ ML for physical simulations, modeling turbulence. 
It is rather challenging to prove that an ML model has arbitrarily learned the laws of physics embedded into the problem 
with the ability to extrapolate within the parameter-space of the simulation. Inability to directly infer the prediction 
capabilities of ML is one of the major causes behind the slow adaption rates, however, the effectiveness of ML cannot be ignored by the community.

Turbulence is ubiquitous in astrophysical environments; however, it involves physics at a vast range of temporal and spatial 
scales, making accurate fully-resolved modeling difficult. Various analytical turbulence models have been developed to be used 
in simulations using time or spatial averaged governing equations, such as in RANS (Reynolds-averaged Navier-Stokes) and LES (Large Eddy Simulation), 
but accuracy is lacking. In search of better methods to model 
turbulence in core-collapse supernovae, it became apparent that ML has great potential 
to produce more accurate turbulence models on an unaveraged subgrid-scale than the current methods. Scientists from both industry and 
academia [@king2016], [@zhang2018] have already begun using ML for applied turbulent problems, but none reach out to 
a theoretical medium of physics and astronomy community on a practical level. For example, physics-based model 
evaluation and interpretability tools are not standardized nor they are widely available. As a result, it is a common 
struggle to verify published results, with the setup not fully disclosed, the code being poorly structured and/or commented or 
even worse - not publicly available; the problem ML community can relate to as well [@Hutson725]. Thus, it is not surprising that there is considerable 
skepticism against ML in physical sciences, with astrophysics being no exception [@carleo2019].

In pursuit of our supernova (SNe) study, the issues outlined above became painfully apparent. Thus, we attempted to simplify 
the barrier to entry for new researchers in domain science fields studying turbulence to employ ML, with the main 
focus on astrophysical applications. As a result, an ML python-based pipeline called ``Sapsan`` has been developed. The goals have 
been to make it accessible and catered to the community through Jupyter Notebooks, command-line-interface (CLI) and graphical-user-interface 
(GUI)\footnote{demo available at [sapsan.app](https://sapsan.app/])} 
available for the end-user. ``Sapsan`` includes built-in optimized ML models for turbulence treatment, both conventional and 
physics-based. More importantly, at its core, the framework is meant to be flexible and modular, hence there is an intuitive 
interface for users to work on their own ML algorithms. In addition, ``Sapsan`` brings best practices from the industry when it 
comes to ML development frameworks. For example, ``Sapsan`` includes docker containers for reproducible release, as well as 
[MLflow](https://mlflow.org/) for experiment tracking.

`Sapsan` is distributed through [Github](https://github.com/pikarpov-LANL/Sapsan) and [pip](https://pypi.org/project/sapsan/). For further reference, [wiki](https://github.com/pikarpov-LANL/Sapsan/wiki) is mainted on Github as well.

# Framework
``Sapsan`` organizes workflow via three repesctive stages: data preparation, machine learning, and analysis. The whole process can be further wrapped in Docker to publish your results for reproducibility, as reflected by Figure 1. Let's brake down each stage in the context of turbulence subgrid modeling, e.g. a model to predict turbulent behavior at the under-resolved simulation scales.

* __Data Preparation__
  * __Loading Data:__ ``Sapsan`` is ready to process common 2D & 3D hydrodynamic (HD) and magnetohydrodynamic (MHD) turbulence data, in simulation-code-specific data formats, such as HDF5 (with more to come per community need).
  * __Transformations:__ a variety of tools are ready for you to ready the data for training
     * __Filter:__ in order to build a subgrid model, one will have to filter the data, e.g. remove small-scale perturbations. 
    A few examples would be either a box, spectral, or a gaussian filter.  The data can be filtered on the fly within the framework.
     * __Sample:__ to run quick tests of your model, you might want to test on a sampled version of the data, wyhile retaining the full spacial domain. Thus, equidistant sampling is ready to go in ``Sapsan``
     * __Batch:__ the pipeline will spatially batch the data
     * __Split:__ data is divided into testing and validation datasets
    
* __Machine Learning__

  * __Model Set Up:__ different ML models would be appropriate for different physical regimes. ``Sapsan`` provides templates for a selection of both conventional and physics-based models with more to come. Only important options are left up to the user to edit, with most of the overhead kept in the backend.
     * __Layers:__  define and order the ML layers
     * __Tracking:__ add extra parameters to be tracked by MLflow
     * __Loos Function:__ decide on a conventional or physics-based/custom loss function to model the data
     * __Optimizer:__ choose how to optimize the training
     * __Scheduler:__ select how to adjust the learning rate

* __Analysis__
  *  __Trained Model:__ a turbulence subgrid model telling us how small-scale structure affects the large scale quantities, i.e. it completes or ''closes'' the governing large-scale equations of motion with small-scale terms. The prediction from a trained ML model is used to provide the needed quantities.
  *  __Analytical Tools:__ compare the trained model with conventional analytic turbulence models, such as Dynamic Smagorisnky [@lilly1966] or Gradient model [@]. Furthermore, conduct other physics-based tests, for example, compute the power spectrum of your prediction.

For further information on each stage please refer to [Sapsan's Wiki on Gihub](https://github.com/pikarpov-LANL/Sapsan/wiki).

![High-level overview of ``Sapsan's`` workflow.](Sapsan_highlevel_overview.png)


# Applications

While ``Sapsan`` is built to be highly customizable to be used in a wide variety of projects in physical sciences, it is optimized for the study of turbulence. 

## Hydro simulations

Here are a few examples of a turbulence closure model, trained on the high-resolution Johns Hopkins Turbulence Database (JHTDB) [@jhtdb2008]. The dataset used in this comparison is a direct numerical simulation (DNS) of statistically-stationary isotropic 3D MHD
turbulence dataset, 1024<sup>3</sup> in spatial resolution and covering roughly one large eddy turnover time over 1024 checkpoints, e.i. dynamical time of the system
[@Eyink2013]. We are comparing it with a commonly used Dynamic Smagorinsky (DS) turbulence closure model [@lilly1966]. On ``Sapsan``  side, a Kernel
Ridge Regression model [@murphy2004] is used to demonstrate the effectiveness of conventional ML approaches in tackling turbulence
problems. In this test we used the following setup:


* __Train features:__ velocity (*u*), vector potential (*A*), magnetic field (*B*), and their respective derivatives at timestep = 1. All quantities have been filtered to remove small-scale perturbations, mimicking the lower fidelity of a non-DNS simulation.
* __Model Input:__ low fidelity velocity (*u*), vector potential (*A*), magnetic field (*B*), and their respective derivatives at a set timestep in the future.
* __Model Output:__ velocity stress tensor ($\tau$) at the matching timestep in the future, which effectively represents the difference between large and small scale structures of the system.

In Figure 2, it can be seen that the ML-based approach significantly outperforms the DS subgrid model in reproducing the probability density function, i.e. statistical distribution of the stress tensor. The results are consistent with [@king2016].

![Predicting turbulent stress-tensor in statistically-stationary isotropic MHD turbulence setup. The outmost left plot compares the original spatial map of the stress-tensor component to the plot in the middle depicting the predicted spatial map. The plot on the right presents probability density functions (PDF), i.e. distributions, of the original stress-tensor component values, the ML predicted values, and the conventional DS subgrid model prediction.](JHTDB.png)

## Supernovae
Even though the conventional regression-based ML approach worked well in the previous section, the complexity of our physical problem forced us to seek a more rigorous ML method. Supernovae host a
different physical regime that is far from the idealistic MHD turbulence case from before. Here we are dealing with dynamically
changing statistics and evolution of the turbulence that is not necessarily isotropic. Depending on the evolutionary stage, turbulence
can behave drastically different, hence a more sophisticated model is required. With ``Sapsan``, we have tested a 3D CNN (Convolutional Neural Network) model 
in attempts to predict a turbulent velocity stress tensor in a realistic Core-Collapse Supernova (CCSN) case. Figure 3 presents results of the following:

* __Train features:__ velocity (*u*), magnetic field (*B*), and their respective derivatives at timestep = 500 (halfway of the total simulation). All quantities have been filtered to remove small-scale perturbations, mimicking the lower fidelity of a non-DNS simulation.
* __Model Input:__ low fidelity velocity (*u*), magnetic field (*B*), and their respective derivatives at a set timestep in the future.
* __Model Output:__ velocity stress tensor ($\tau$) at the matching timestep in the future, which effectively represents the difference between large and small scale structures of the system.

In this case, the matching level of the distributions is the most important factor. It can be seen that the probability density functions match quite closely, with the outlier at the most negative values being the exception, even though the prediction is done far into the future (timestep=1000, end of the simulation time). For further discussion on the comparison of the models and the results, please refer to [the ApJ paper].

![Predicting Turbulent Stress-Tensor in a Core-Collapse Supernovae (CCSN). The model has been trained on a 3D MHD direct numerical simulation (DNS) of the first 10ms after the shockwave has bounced off of the core in a CCSN scenario. On the left, the two figures are the 2D slices of a 3D data-cube prediction, with the right plot presenting a comparison of PDFs of the original 3D data, 3D ML prediction, and a conventional Gradient subgrid model.](ccsn_mri_plots.png)

# Acknowledgements
The development of ``Sapsan`` was supported by the Laboratory Directed Research and Development program and the Center for Space and Earth Science at Los Alamos National Laboratory through the student fellow grant. In addition, We would like to thank DOE SciDAC for additional funding support.

# References
