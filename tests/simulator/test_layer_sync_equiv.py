# This a high-level test to show that if F == number of neurons per layer,
# then the outcome of the simulation is the same as the outcome of the
# simulation of the equivalent layer synchronized network.

import torch
from asynctorch.nn.architecture.mixed_architecture import AsyncNetwork, MixedArchitecture
from asynctorch.nn.neuron.lif_state import LIFState
import torch.nn as nn
import numpy as np
from asynctorch.simulator.async_simulator import AsyncSimulator

from asynctorch.simulator.spike_scheduler import RandomSpikeScheduler
from asynctorch.simulator.spike_selector import SpikeSelector
from asynctorch.utils.surrogate import ATan
from .leaky import Leaky

### Linear layer test
def build_linear_layer_synchronized_module(tau_m, threshold, weights_linear1, weights_linear2, device) -> nn.Module:
    beta = np.exp(-1.0 / tau_m)
    linear1 = torch.nn.Linear(3, 3, bias=False)
    linear1.weight.data = weights_linear1
    linear2 = torch.nn.Linear(3, 2, bias=False)
    linear2.weight.data = weights_linear2
    return torch.nn.Sequential(
        linear1,
        Leaky(beta=beta, threshold=threshold),
        linear2,
        Leaky(beta=beta, threshold=threshold, output=True),
    )

def build_linear_async_torch_module(tau_m, threshold, weights_linear1, weights_linear2, device) -> AsyncSimulator:
    spike_grad = ATan()
    neurons_per_layer = [3, 2]
    n_inputs = 3
    n_neurons = sum(neurons_per_layer)
    async_network: AsyncNetwork = AsyncNetwork.build_sequential([nn.Linear(n_inputs, 3, bias=False), nn.Linear(3, 2, bias=False)], [(n_inputs,), (3,), (2,)])
    async_network.input_layer.module.weight.data = weights_linear1
    async_network.layers[0].module.weight.data = weights_linear2
    tau_m = torch.full((n_neurons,), tau_m, dtype=torch.float32)
    membrane_threshold = torch.full((n_neurons,), threshold, dtype=torch.float32)
    state_module = LIFState(neurons_per_layer, tau_m, membrane_threshold, spike_grad, device)
    forward_module = MixedArchitecture(async_network, device)
    spike_scheduler = RandomSpikeScheduler(state_module)
    spike_selector_module = SpikeSelector(forward_module, spike_scheduler, 3, device, prioritize_input=True)
    return AsyncSimulator(state_module, spike_selector_module)

def test_linear_layer_sync_equiv():
    device = torch.device("cpu")
    tau_m = 1.0
    threshold = 1.0
    weights_linear1 = torch.randn(3, 3, dtype=torch.float32)
    weights_linear2 = torch.randn(2, 3, dtype=torch.float32)
    layer_sync_module = build_linear_layer_synchronized_module(tau_m, threshold, weights_linear1, weights_linear2, device)
    async_torch_module = build_linear_async_torch_module(tau_m, threshold, weights_linear1, weights_linear2, device)
    n_test_iterations = 1000
    for i in range(n_test_iterations):
        input_spks = torch.ones((1,3), dtype=torch.float32)
        layer_sync_output, mem = layer_sync_module(input_spks)
        async_torch_output = async_torch_module(input_spks, dt=1.0)
        assert async_torch_module.spike_selector.is_done()
        assert torch.allclose(layer_sync_output, async_torch_output[0, -2:], atol=1e-6)


### Convolutional layer test
def build_conv_layer_synchronized_module(tau_m, threshold, weights_conv1, weights_conv2, weights_fc, device) -> nn.Module:
    beta = np.exp(-1.0 / tau_m)
    conv1 = torch.nn.Conv2d(2, 3, kernel_size=3, stride=2, padding=1, bias=False, device=device)
    conv1.weight.data = weights_conv1
    conv2 = torch.nn.Conv2d(3, 2, kernel_size=3, stride=2, padding=1, bias=False, device=device)
    conv2.weight.data = weights_conv2
    fc = torch.nn.Linear(32, 2, bias=False, device=device)
    fc.weight.data = weights_fc
    return torch.nn.Sequential(
        conv1,
        Leaky(beta=beta, threshold=threshold),
        conv2,
        Leaky(beta=beta, threshold=threshold),
        torch.nn.Flatten(),
        fc,
        Leaky(beta=beta, threshold=threshold, output=True),
    )

def build_conv_async_torch_module(tau_m, threshold, weights_conv1, weights_conv2, weights_fc, device) -> AsyncSimulator:
    spike_grad = ATan()
    shape_per_layer = [(2, 16, 16), (3, 8, 8), (32,), (2,)]
    neurons_per_layer = [np.prod(shape) for shape in shape_per_layer[1:]]
    async_network: AsyncNetwork = AsyncNetwork.build_sequential([
            nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Linear(32, 2, bias=False)
        ], 
        shape_per_layer
    )
    async_network.input_layer.module.weight.data = weights_conv1
    async_network.layers[0].module.weight.data = weights_conv2
    async_network.layers[1].module.weight.data = weights_fc
    forward_module = MixedArchitecture(async_network, device)
    state_module = LIFState(neurons_per_layer, tau_m, threshold, spike_grad, device)
    spike_scheduler = RandomSpikeScheduler(state_module)
    spike_selector_module = SpikeSelector(forward_module, spike_scheduler, 192, device, prioritize_input=True)
    return AsyncSimulator(state_module, spike_selector_module)

def test_conv_layer_sync_equiv():
    device = torch.device("cpu")
    tau_m = 1.0
    threshold = 1.0
    weights_conv1 = torch.randn(3, 2, 3, 3, dtype=torch.float32)
    weights_conv2 = torch.randn(2, 3, 3, 3, dtype=torch.float32)
    weights_fc = torch.randn(2, 32, dtype=torch.float32)
    layer_sync_module = build_conv_layer_synchronized_module(tau_m, threshold, weights_conv1, weights_conv2, weights_fc, device)
    async_torch_module = build_conv_async_torch_module(tau_m, threshold, weights_conv1, weights_conv2, weights_fc, device)
    n_test_iterations = 1000
    for i in range(n_test_iterations):
        input_spks = torch.ones((1,2,16,16), dtype=torch.float32)
        layer_sync_output, mem = layer_sync_module(input_spks)
        async_torch_output = async_torch_module(input_spks.reshape(1, -1), dt=1.0)
        assert async_torch_module.spike_selector.is_done()
        assert torch.allclose(layer_sync_output, async_torch_output[0, -2:], atol=1e-6)
