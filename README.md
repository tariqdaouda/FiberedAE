# FiberedAE
Implementation of Fibered Auto-Encoders (FAEs)
FAE are Auto Encoders whose space of latent variables have a fiber bundle structure. This implementation acts as a proof of concept and aims at illustrating the concept.
    
This repository contains all code for creating and training FAEs.

  * Examples in the form of Jupyter notebooks are available in the the folder: **/demos**.
  * Model hyper-parameters can be found in a human readable format in: **/demos/configuration**.

## Installation

This package has only been tested on Ubuntu machines running python 3.7. We highly recommend installing the package in a fresh Anaconda environment.

To create the environment we used:

```conda create -n fae python=3.7```

Then to activate it:

```conda activate fae```

To install the package go inside the package folder and install it using:

```python setup.py develop```

If you are running miniconda and want to run the notebooks, you further need to install jupyter lab:

```conda install -c conda-forge jupyterlab```

You may also need to install ipywidgets for progress bars to work.

```conda install -c conda-forge ipywidgets```

## Running it

After installation a CLI will be available for training, integration etc.... run:

```fae --help```

for more details.
