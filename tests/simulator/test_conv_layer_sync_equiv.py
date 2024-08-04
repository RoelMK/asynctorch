import torch
from snntorch import surrogate
from asynctorch.nn.architecture.conv2d_architecture import Conv2dArchitecture, Conv2dParams
from asynctorch.nn.architecture.sparse_fully_linear_architecture import SparseFullyLinearArchitecture
from asynctorch.nn.neuron.lif_state import LIFState
import torch.nn as nn
import snntorch as snn
import numpy as np
from asynctorch.simulator.async_simulator import AsyncSimulator

from asynctorch.simulator.spike_scheduler import RandomSpikeScheduler
from asynctorch.simulator.spike_selector import SpikeSelector
from leaky import Leaky

        
def build_layer_synchronized_module(tau_m, threshold, weights_conv1, weights_conv2, weights_fc, device) -> nn.Module:
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

def build_async_torch_module(tau_m, threshold, weights_conv1, weights_conv2, weights_fc, device) -> AsyncSimulator:
    spike_grad = surrogate.atan()
    neurons_per_layer = [192, 32, 2]
    conv_params_per_layer = [
        Conv2dParams(in_channels=2, out_channels=3, kernel_size=3, stride=2, padding=1),
        Conv2dParams(in_channels=3, out_channels=2, kernel_size=3, stride=2, padding=1),
    ]
    n_neurons = sum(neurons_per_layer)
    tau_m = torch.full((n_neurons,), tau_m, dtype=torch.float32)
    membrane_threshold = torch.full((n_neurons,), threshold, dtype=torch.float32)
    sync_threshold = torch.zeros(n_neurons, dtype=torch.float32)
    state_module = LIFState(neurons_per_layer, tau_m, membrane_threshold, sync_threshold, spike_grad, spike_grad, device)
    forward_module = Conv2dArchitecture(2, 16, 16, neurons_per_layer, conv_params_per_layer, device)
    state_dict = Conv2dArchitecture.create_state_dict([weights_conv1, weights_conv2], weights_fc)
    forward_module.load_state_dict(state_dict)
    spike_scheduler = RandomSpikeScheduler(state_module)
    spike_selector_module = SpikeSelector(forward_module, spike_scheduler, 192, device, prioritize_input=True)
    return AsyncSimulator(state_module, spike_selector_module)

def test_layer_sync_equiv():
    device = torch.device("cpu")
    tau_m = 1.0
    threshold = 1.0
    weights_conv1 = torch.randn(3, 2, 3, 3, dtype=torch.float32)
    weights_conv2 = torch.randn(2, 3, 3, 3, dtype=torch.float32)
    weights_fc = torch.randn(2, 32, dtype=torch.float32)
    layer_sync_module = build_layer_synchronized_module(tau_m, threshold, weights_conv1, weights_conv2, weights_fc, device)
    async_torch_module = build_async_torch_module(tau_m, threshold, weights_conv1, weights_conv2, weights_fc, device)
    n_test_iterations = 1000
    for i in range(n_test_iterations):
        input_spks = torch.ones((1,2,16,16), dtype=torch.float32)
        layer_sync_output, mem = layer_sync_module(input_spks)
        async_torch_output = async_torch_module(input_spks, dt=1.0)
        assert async_torch_module.spike_selector.is_done()
        assert torch.allclose(layer_sync_output, async_torch_output[0, -2:], atol=1e-6)
