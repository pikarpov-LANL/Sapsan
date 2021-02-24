# Sapsan  <a href="http://sapsan.app"><img src="https://github.com/pikarpov-LANL/Sapsan/blob/master/docs/images/logo3_black_slim_notitle_whitebg.png?raw=true"  alt="Sapsan logo" align="right" width="100"></a>

Sapsan is a pipeline for easy Machine Learning implementation in scientific projects. That being said, its primary goal and featured models are geared towards dynamic MHD turbulence subgrid modeling. Sapsan will soon feature Physics-Informed Machine Learning models in its set of tools to accurately capture the turbulent nature applicable to Core-Collapse Supernovae.

Feel free to check out a website version at [sapsan.app](http://sapsan.app). The interface is indentical to the GUI of the local version of Sapsan, except lacking the ability to edit the model code on the fly.

Note: currently Sapsan is in alpha, but we are actively working on it and introduce new feature on a daily basis.

## [Sapsan's Wiki](https://github.com/pikarpov-LANL/Sapsan/wiki)

Please refer to Sapsan's github wiki to learn more about framework's details and capabilities.

## Quick Start

#### 1. Clone from git (recommended)
```shell script
git clone https://github.com/pikarpov-LANL/Sapsan.git
cd Sapsan/
python setup.py install
```

For **GPU** enabled version change the last line to
```shell script
python setup_gpu.py install
```

#### 2. Install via pip (cpu-only)
```shell script
pip install sapsan
```

Note: make sure you are using the latest release version

#### Run Examples

To make sure everything is alright and to familiarize yourself with the interface, please run the following CNN example on 3D data:
```shell script
jupyter notebook sapsan/examples/cnn_example.ipynb
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
