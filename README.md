# Introduction
**This is an alpha release. The documentation and package are still under construction.**

Asynctorch is a PyTorch extension that allows training and inference of spiking neural networks (SNNs) using network asynchrony. It is built on top of PyTorch and snnTorch and is designed to be easy to use for researchers and developers familiar with PyTorch.

# Installation
Asynctorch is not yet on PyPI. At this moment, only installation from the source is supported.

To install directly from the source:
```console
git clone https://github.com/RoelMK/asynctorch
cd asynctorch
pip install -e .
```

# Tutorials
**The notebooks are currently work-in-progress! This information is outdated.**

You can start by checking the examples in the `examples` folder. The examples are written in Jupyter notebooks and can be run in a Jupyter notebook environment. 

~~| Notebook | Content |
|---|---|
| tutorial_training.ipynb | Train a model using asynctorch |
| tutorial_inference.ipynb | Infer using both a model trained with snnTorch and a model trained with asynctorch |~~


# Structure
The repository is structured as follows:

| Package | Content |
|---|---|
| asynctorch.nn.architecture | Architectures (currently supported: mixed) |
| asynctorch.nn.neuron | Neuron models (currently supported: LIF) |
| asynctorch.simulator | Simulator base modules |
| asynctorch.simulator.extensions |  Extensions to the simulator |

Below is a description of the supported architectures, neuron models, and simulator extensions. Additional information can be found in the docstrings of the classes (or will be added in the future).


## Architectures
The following architectures are supported:
| Class | Type | Description |
|---|---|---|
| MixedArchitecture | Any type | Build sophisticated networks by connecting torch.nn modules. Does not support unprioritized input yet. |


## Neuron models
The following neuron models are supported:
| Class | Type | Description |
|---|---|---|
| LIFState | LIF | Basic LIF neuron model, also includes support for limiting spiking per timestep (refractoriness), including refractory dropout, and synchronization thresholds. |


## Simulator extensions
Extensions can manipulate and/or log spike/current activity. If an extension only logs activity, it is a monitor. The following extensions are supported:
| Class | Description |
|---|---|
| EarlyStoppingExtension | Stops inference after a fixed number of forward steps. |
| SpikeDropoutExtension | Randomly drops spikes. Can be applied to input spikes only, network spikes only, or both. |
| StopOnOutputExtension | Stops inference when the output layer spikes. Optionally, the stopping can be delayed by a fixed number of forward steps. |
| SpikesToOutputMonitor | Monitors the number of spikes in the output layer. |
| CurrentsToSpikeMonitor | Monitors the number of spikes required for a neuron to spike. |