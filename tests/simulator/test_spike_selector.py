from asynctorch.nn.architecture.mixed_architecture import AsyncNetwork, MixedArchitecture
from asynctorch.simulator.spike_scheduler import RandomSpikeScheduler
from asynctorch.simulator.spike_selector import SpikeSelector
import torch
from torch import nn

def test_zero_weights_forward():
    device = torch.device("cpu")
    async_network = AsyncNetwork.build_sequential([nn.Linear(10, 10, bias=False)], [(10,), (10,)])
    forward_module = MixedArchitecture(async_network, device)
    spike_scheduler = RandomSpikeScheduler(None)
    spike_selector_module = SpikeSelector(forward_module, spike_scheduler, 2, device, prioritize_input=True)
    assert not spike_selector_module.is_init()
    spike_selector_module._init_state(2)
    assert spike_selector_module.is_init()
    assert spike_selector_module.is_done()
    s_in = torch.zeros((2, 10), device=device)
    currents, n_currents = spike_selector_module(s_in, is_input=False)
    assert spike_selector_module.is_done()
    assert currents.shape == (2, 10)
    assert n_currents.shape == (2, 10)
    assert currents.sum() == 0
    assert n_currents.sum() == 0

def test_prioritized_input_forward():
    device = torch.device("cpu")
    async_network = AsyncNetwork.build_sequential([nn.Linear(2, 3, bias=False), nn.Linear(3, 4, bias=False)], [(2,), (3,), (4,)])
    async_network.input_layer.module.weight.data = torch.ones((3, 2), dtype=torch.float32)
    async_network.layers[0].module.weight.data = torch.ones((4, 3), dtype=torch.float32)
    forward_module = MixedArchitecture(async_network, device)
    spike_scheduler = RandomSpikeScheduler(None)
    spike_selector_module = SpikeSelector(forward_module, spike_scheduler, 2, device, prioritize_input=True)
    spike_selector_module._init_state(2)
    s_in = torch.ones((2, 2), device=device)
    currents, n_currents = spike_selector_module(s_in, is_input=True)
    assert currents.sum() == 12
    assert spike_selector_module.outgoing_spikes.sum() == 0

def test_neuron_activity_forward_prioritized_input():
    device = torch.device("cpu")
    async_network = AsyncNetwork.build_sequential([nn.Linear(3, 3, bias=False), nn.Linear(3, 3, bias=False)], [(3,), (3,), (3,)])
    async_network.input_layer.module.weight.data = torch.ones((3, 3), dtype=torch.float32)
    async_network.layers[0].module.weight.data = torch.ones((3, 3), dtype=torch.float32)
    forward_module = MixedArchitecture(async_network, device)
    spike_scheduler = RandomSpikeScheduler(None)
    spike_selector_module = SpikeSelector(forward_module, spike_scheduler, 2, device, prioritize_input=True)
    spike_selector_module._init_state(2)
    s_in = torch.asarray([[1,1,1,0,0,0],[1,1,1,0,0,0]], dtype=torch.float32, device=device)
    currents, n_currents = spike_selector_module(s_in, is_input=False)
    assert currents.sum() == 12
    assert spike_selector_module.outgoing_spikes.sum() == 2
    assert spike_selector_module.outgoing_spikes[0].sum() == 1
    assert not spike_selector_module.is_done()