# Sapsan  <a href="http://sapsan.app"><img src="https://github.com/pikarpov-LANL/Sapsan/blob/images/docs/images/logo3_black_slim_notitle_whitebg.png?raw=true"  alt="Sapsan logo" align="right" width="100"></a>

Sapsan is a pipeline for Machine Learning (ML) based turbulence modeling. While turbulence is important in a wide range of mediums, the pipeline primarily focuses on astrophysical applications. With Sapsan, one can create their own custom models or use either conventional or physics-informed ML approaches for turbulence modeling included with the pipeline ([estimators](https://github.com/pikarpov-LANL/Sapsan/wiki/Estimators)). Sapsan is designed to take out all the hard work from data preparation and analysis, leaving you focused on ML model design, layer by layer.

Feel free to check out a website version at [sapsan.app](http://sapsan.app). The interface is identical to the GUI of the local version of Sapsan, except lacking the ability to edit the model code on the fly.

## [Sapsan's Wiki](https://github.com/pikarpov-LANL/Sapsan/wiki)

Please refer to Sapsan's Github Wiki to learn more about the framework's details and capabilities.

## Quick Start

### 1. Install PyTorch (prerequisite)
Sapsan can be run on both CPU and GPU. Please follow the instructions on [PyTorch](https://pytorch.org/get-started/locally/) to install the latest version (torch>=1.7.1 & CUDA>=11.0).

### 2. Install via pip (recommended)
```
pip install sapsan
```

#### OR Clone from git
```
git clone https://github.com/pikarpov-LANL/Sapsan.git
cd Sapsan/
python setup.py install
```

Note: see [Installation Page](https://github.com/pikarpov-LANL/Sapsan/wiki/Installation/) on the Wiki for complete instructions with Graphviz and Docker installation.

### 3. Test Installation

To make sure everything is alright, run a test of your setup:
```
sapsan test
```

### 4. Run Examples

To get started and familiarize yourself with the interface, feel free to run the included examples ([CNN](https://github.com/pikarpov-LANL/Sapsan/wiki/Estimators#convolution-neural-network-cnn) or [PICAE](https://github.com/pikarpov-LANL/Sapsan/wiki/Estimators#physics-informed-convolutional-autoencoder-picae) on 3D data, and [KRR](https://github.com/pikarpov-LANL/Sapsan/wiki/Estimators#kernel-ridge-regression-krr) on 2D data). To copy the examples, type:
```
sapsan get_examples
```
This will create a folder `./sapsan_examples` with appropriate example jupyter notebooks.

### 5. Create Custom Projects!
To start a custom project, designing your own custom estimator, i.e., network, go ahead and run:
```
sapsan create {name}
```
where `{name}` should be replaced with your custom project name. As a result, a pre-filled template for the estimator, a jupyter notebook to run everything from, and Docker will be initialized.




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
