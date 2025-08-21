from typing import List, Tuple
import torch
from asynctorch.nn.architecture.mixed_architecture import AsyncNetwork, MixedArchitecture
from asynctorch.nn.neuron.lif_state import LIFState
from asynctorch.simulator.async_simulator import AsyncSimulator
from asynctorch.simulator.spike_scheduler import RandomSpikeScheduler
from asynctorch.simulator.spike_selector import SpikeSelector
from torch import nn
from math import prod

from asynctorch.utils.surrogate import ATan

def build_simple_async_simulator(fill_weight: float = None, *args, **kwargs):
    modules = [nn.Linear(2, 3, bias=False), nn.Linear(3, 4, bias=False)]
    if fill_weight is not None:
        for module in modules:
            module.weight.data.fill_(fill_weight)
    return build_async_simulator(modules, [(2,), (3,), (4,)], *args, **kwargs)

def build_async_simulator(module_per_layer: List[nn.Module], shape_per_layer: List[Tuple[int, ...]], 
                          tau_m=1.0, membrane_threshold=1.0, forward_group_size=2, forward_step_extensions=[], apply_refrac=True):
    device = torch.device("cpu")
    neurons_per_layer = [prod(shape) for shape in shape_per_layer[1:]]
    async_network = AsyncNetwork.build_sequential(module_per_layer, shape_per_layer)
    spike_grad = ATan(2)

    state_module = LIFState(neurons_per_layer, tau_m, membrane_threshold, spike_grad, device, apply_refrac=apply_refrac)
    forward_module = MixedArchitecture(async_network, device)
    spike_scheduler = RandomSpikeScheduler(None)
    spike_selector_module = SpikeSelector(forward_module, spike_scheduler, forward_group_size, device, prioritize_input=True)
    return AsyncSimulator(state_module, spike_selector_module, forward_step_extensions=forward_step_extensions)

def test_initialize_async_simulator():
    async_simulator = build_simple_async_simulator()
    assert not async_simulator.is_init()
    async_simulator.init_state(batch_size=2)
    assert async_simulator.is_init()
    async_simulator.reset_state()
    assert not async_simulator.is_init()
    async_simulator.init_state(batch_size=2)
    assert async_simulator.is_init()

def test_async_simulator_one_forward():
    async_simulator = build_simple_async_simulator(fill_weight=1.0)
    async_simulator.init_state(batch_size=2)
    s_in = torch.ones((2, 2))
    spike_counts = async_simulator(s_in, dt=1.0)
    assert spike_counts.shape == (2, 7)
    assert spike_counts.sum() == 14
    spike_counts = async_simulator(s_in, dt=1.0)
    assert spike_counts.sum() == 14
    assert async_simulator.spike_counts.sum() == 28

def test_async_simulator_zero_forward():
    async_simulator = build_simple_async_simulator(fill_weight=1.0)
    async_simulator.init_state(batch_size=2)
    s_in = torch.zeros((2, 2))
    spike_counts = async_simulator(s_in, dt=1.0)
    assert spike_counts.shape == (2, 7)
    assert spike_counts.sum() == 0
    spike_counts = async_simulator(s_in, dt=1.0)
    assert spike_counts.sum() == 0
    assert async_simulator.spike_counts.sum() == 0

def test_async_simulator_forward_increased_threshold():
    async_simulator = build_simple_async_simulator(fill_weight=1.0, membrane_threshold=3.5)
    async_simulator.init_state(batch_size=2)
    s_in = torch.ones((2, 2))
    spike_counts = async_simulator(s_in, dt=0.0)
    assert spike_counts.shape == (2, 7)
    assert spike_counts.sum() == 0
    spike_counts = async_simulator(s_in, dt=0.0)
    assert spike_counts.sum() == 6 # first layer
    assert async_simulator.spike_counts.sum() == 6
    spike_counts = async_simulator(s_in, dt=0.0)
    assert spike_counts.sum() == 0
    spike_counts = async_simulator(s_in, dt=0.0)
    assert spike_counts.sum() == 14 # first and second layer
    assert async_simulator.spike_counts.sum() == 20
    spike_counts = async_simulator(s_in, dt=0.0)
    assert spike_counts.sum() == 0
    async_simulator.reset_state()
    async_simulator.init_state(batch_size=2)
    assert async_simulator.spike_counts.sum() == 0

def test_async_simulator_forward_decay_membrane_potential():
    async_simulator = build_simple_async_simulator(fill_weight=1.0, membrane_threshold=2.0, tau_m=0.00000001)
    async_simulator.init_state(batch_size=2)
    s_in = torch.ones((2, 2))
    spike_counts = async_simulator(s_in, dt=1.0)
    assert spike_counts.shape == (2, 7)
    assert spike_counts.sum() == 0
    spike_counts = async_simulator(s_in, dt=1.0)
    assert spike_counts.sum() == 0
    spike_counts = async_simulator(s_in, dt=1.0)
    assert spike_counts.sum() == 0
