# This a high-level test to show that if F == number of neurons per layer,
# then the outcome of the simulation is the same as the outcome of the
# simulation of the equivalent layer synchronized network.

import torch
from snntorch import surrogate
from asynctorch.nn.architecture.mixed_architecture import AsyncNetwork, MixedArchitecture
from asynctorch.nn.neuron.lif_state import LIFState
import torch.nn as nn
import snntorch as snn
import numpy as np
from asynctorch.simulator.async_simulator import AsyncSimulator

from asynctorch.simulator.spike_scheduler import RandomSpikeScheduler
from asynctorch.simulator.spike_selector import SpikeSelector
from .leaky import Leaky

def build_layer_synchronized_module(tau_m, threshold, weights_linear1, weights_linear2, device) -> nn.Module:
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

def build_async_torch_module(tau_m, threshold, weights_linear1, weights_linear2, device) -> AsyncSimulator:
    spike_grad = surrogate.atan()
    neurons_per_layer = [3, 2]
    n_inputs = 3
    n_neurons = sum(neurons_per_layer)
    async_network: AsyncNetwork = AsyncNetwork.build_sequential([nn.Linear(n_inputs, 3, bias=False), nn.Linear(3, 2, bias=False)], [(n_inputs,), (3,), (2,)])
    async_network.input_layer.module.weight.data = weights_linear1
    async_network.layers[0].module.weight.data = weights_linear2
    tau_m = torch.full((n_neurons,), tau_m, dtype=torch.float32)
    membrane_threshold = torch.full((n_neurons,), threshold, dtype=torch.float32)
    sync_threshold = torch.zeros(n_neurons, dtype=torch.float32)
    state_module = LIFState(neurons_per_layer, tau_m, membrane_threshold, sync_threshold, spike_grad, spike_grad, device)
    forward_module = MixedArchitecture(async_network, device)
    spike_scheduler = RandomSpikeScheduler(state_module)
    spike_selector_module = SpikeSelector(forward_module, spike_scheduler, 3, device, prioritize_input=False)
    return AsyncSimulator(state_module, spike_selector_module)

def test_layer_sync_equiv():
    device = torch.device("cpu")
    tau_m = 1.0
    threshold = 1.0
    weights_linear1 = torch.randn(3, 3, dtype=torch.float32)
    weights_linear2 = torch.randn(2, 3, dtype=torch.float32)
    layer_sync_module = build_layer_synchronized_module(tau_m, threshold, weights_linear1, weights_linear2, device)
    async_torch_module = build_async_torch_module(tau_m, threshold, weights_linear1, weights_linear2, device)
    n_test_iterations = 1000
    for i in range(n_test_iterations):
        input_spks = torch.ones((1,3), dtype=torch.float32)
        layer_sync_output, mem = layer_sync_module(input_spks)
        async_torch_output = async_torch_module(input_spks, dt=1.0)
        assert async_torch_module.spike_selector.is_done()
        assert torch.allclose(layer_sync_output, async_torch_output[0, -2:], atol=1e-6)
