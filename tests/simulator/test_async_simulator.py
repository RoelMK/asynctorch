import torch
from snntorch import surrogate
from asynctorch.nn.architecture.mixed_architecture import AsyncNetwork, MixedArchitecture
from asynctorch.nn.neuron.lif_state import LIFState
from asynctorch.simulator.async_simulator import AsyncSimulator
from asynctorch.simulator.spike_scheduler import RandomSpikeScheduler
from asynctorch.simulator.spike_selector import SpikeSelector
from torch import nn


def build_async_simulator(tau_m=1.0, membrane_threshold=1.0, sync_threshold=0.0, forward_group_size=2, forward_step_extensions=[], apply_refrac=True, prioritize_input=False):
    device = torch.device("cpu")
    n_inputs = 2
    n_neurons = 7
    neurons_per_layer = [3, 4]
    async_network = AsyncNetwork.build_sequential([nn.Linear(n_inputs, 3), nn.Linear(3, 4)], [(n_inputs,), (3,), (4,)])
    tau_m = torch.full((n_neurons,), tau_m, dtype=torch.float32, device=device)
    membrane_threshold = torch.full((n_neurons,), membrane_threshold, dtype=torch.float32, device=device)
    sync_threshold = torch.full((n_neurons,), sync_threshold, dtype=torch.float32, device=device)
    spike_grad = surrogate.atan()

    state_module = LIFState(neurons_per_layer, tau_m, membrane_threshold, sync_threshold, spike_grad, spike_grad, device, apply_refrac=apply_refrac)
    forward_module = MixedArchitecture(async_network, device)
    spike_scheduler = RandomSpikeScheduler(None)
    spike_selector_module = SpikeSelector(forward_module, spike_scheduler, forward_group_size, device, prioritize_input=prioritize_input)
    return AsyncSimulator(state_module, spike_selector_module, forward_step_extensions=forward_step_extensions)

def test_initialize_async_simulator():
    async_simulator = build_async_simulator()
    assert not async_simulator.is_init()
    async_simulator.init_state(batch_size=2)
    assert async_simulator.is_init()
    async_simulator.reset_state()
    assert not async_simulator.is_init()
    async_simulator.init_state(batch_size=2)
    assert async_simulator.is_init()

def test_async_simulator_one_forward():
    async_simulator = build_async_simulator()
    async_simulator.init_state(batch_size=2)
    s_in = torch.ones((2, 2))
    spike_counts = async_simulator(s_in, dt=1.0)
    assert spike_counts.shape == (2, 7)
    assert spike_counts.sum() == 14
    spike_counts = async_simulator(s_in, dt=1.0)
    assert spike_counts.sum() == 14
    assert async_simulator.spike_counts.sum() == 28

def test_async_simulator_zero_forward():
    async_simulator = build_async_simulator()
    async_simulator.init_state(batch_size=2)
    s_in = torch.zeros((2, 2))
    spike_counts = async_simulator(s_in, dt=1.0)
    assert spike_counts.shape == (2, 7)
    assert spike_counts.sum() == 0
    spike_counts = async_simulator(s_in, dt=1.0)
    assert spike_counts.sum() == 0
    assert async_simulator.spike_counts.sum() == 0

def test_async_simulator_forward_increased_threshold():
    async_simulator = build_async_simulator(membrane_threshold=2.0)
    async_simulator.init_state(batch_size=2)
    s_in = torch.ones((2, 2))
    spike_counts = async_simulator(s_in, dt=1.0)
    assert spike_counts.shape == (2, 7)
    assert spike_counts.sum() == 0
    spike_counts = async_simulator(s_in, dt=1.0)
    assert spike_counts.sum() == 14
    assert async_simulator.spike_counts.sum() == 14
    spike_counts = async_simulator(s_in, dt=1.0)
    assert spike_counts.sum() == 0
    spike_counts = async_simulator(s_in, dt=1.0)
    assert spike_counts.sum() == 14
    assert async_simulator.spike_counts.sum() == 28
    spike_counts = async_simulator(s_in, dt=1.0)
    assert spike_counts.sum() == 0
    async_simulator.reset_state()
    async_simulator.init_state(batch_size=2)
    assert async_simulator.spike_counts.sum() == 0

if __name__ == "__main__":
    test_async_simulator_forward_increased_threshold()

def test_async_simulator_forward_decay_membrane_potential():
    async_simulator = build_async_simulator(membrane_threshold=2.0, tau_m=0.00000001)
    async_simulator.init_state(batch_size=2)
    s_in = torch.ones((2, 2))
    spike_counts = async_simulator(s_in, dt=1.0)
    assert spike_counts.shape == (2, 7)
    assert spike_counts.sum() == 0
    spike_counts = async_simulator(s_in, dt=1.0)
    assert spike_counts.sum() == 0
    spike_counts = async_simulator(s_in, dt=1.0)
    assert spike_counts.sum() == 0
