# Sapsan  <a href="http://sapsan.app"><img src="https://github.com/pikarpov-LANL/Sapsan/blob/images/docs/images/logo3_black_slim_notitle_whitebg.png?raw=true"  alt="Sapsan logo" align="right" width="100"></a>

Sapsan is a pipeline for Machine Learning (ML) based turbulence modeling. While turbulence is important in a wide range of mediums, the pipeline primarily focuses on astrophysical application. With Sapsan, one can create their own custom models or use either conventional or physics-informed ML approaches for turbulence modeling included with the pipeline ([estimators](https://github.com/pikarpov-LANL/Sapsan/wiki/Estimators)). Sapsan is designed to take out all the hard work from data preparation and analysis, leaving you focused on ML model design, layer by layer.

Feel free to check out a website version at [sapsan.app](http://sapsan.app). The interface is indentical to the GUI of the local version of Sapsan, except lacking the ability to edit the model code on the fly.

## [Sapsan's Wiki](https://github.com/pikarpov-LANL/Sapsan/wiki)

Please refer to Sapsan's github wiki to learn more about framework's details and capabilities.

## Quick Start

#### 1. Install PyTorch (prerequisite)
Sapsan can be run on both cpu and gpu. Please follow the instructions on [PyTorch](https://pytorch.org/get-started/locally/) to install the latest version (torch>=1.7.1 & CUDA>=11.0).

#### 2. Clone from git (recommended)
```shell script
git clone https://github.com/pikarpov-LANL/Sapsan.git
cd Sapsan/
python setup.py install
```

#### OR Install via pip
```shell script
pip install sapsan
```

Note: see [Installation Page](https://github.com/pikarpov-LANL/Sapsan/wiki/Installation/) on the Wiki for complete instructions with Graphviz and Docker installation.

#### Run Examples

To make sure everything is alright and to familiarize yourself with the interface, please run the following CNN example on 3D data:
```shell script
jupyter notebook sapsan/examples/cnn_example.ipynb
```
alternatively, you can try out the physics-informed convolutional auto-encoder (PICAE) example on random 3D data:
```shell script
jupyter notebook sapsan/examples/picae_example.ipynb
```
or a KRR example on 2D data:
```shell script
jupyter notebook sapsan/examples/krr_example.ipynb
```




-------
Sapsan has a BSD-style license, as found in the [LICENSE](https://github.com/pikarpov-LANL/Sapsan/blob/master/LICENSE) file.

© (or copyright) 2019. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.
