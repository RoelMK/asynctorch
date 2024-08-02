from asynctorch.nn.architecture.sparse_fully_linear_architecture import SparseFullyLinearArchitecture
from asynctorch.simulator.spike_scheduler import RandomSpikeScheduler
from asynctorch.simulator.spike_selector import SpikeSelector
import torch

def test_zero_weights_forward():
    device = torch.device("cpu")
    forward_module = SparseFullyLinearArchitecture(n_inputs=10, neurons_per_layer=[10], device=device)
    spike_scheduler = RandomSpikeScheduler(None)
    spike_selector_module = SpikeSelector(forward_module, spike_scheduler, 2, device, prioritize_input=False)
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

def test_one_weights_input_forward():
    device = torch.device("cpu")
    forward_module = SparseFullyLinearArchitecture(n_inputs=10, neurons_per_layer=[10], device=device)
    weights_per_layer = [
        torch.ones((10, 10), dtype=torch.float32),
    ]
    state_dict = SparseFullyLinearArchitecture.create_state_dict_from_weights_list(weights_per_layer)
    forward_module.load_state_dict(state_dict)
    spike_scheduler = RandomSpikeScheduler(None)
    spike_selector_module = SpikeSelector(forward_module, spike_scheduler, 2, device, prioritize_input=False)
    spike_selector_module._init_state(2)
    s_in = torch.zeros((2, 10), device=device)
    currents, n_currents = spike_selector_module(s_in, is_input=True)
    assert currents.sum() == 0
    assert n_currents.sum() == 0
    s_in = torch.ones((2, 10), device=device)
    currents, n_currents = spike_selector_module(s_in, is_input=True)
    assert currents.sum() == 40
    assert n_currents.sum() == 40
    spike_selector_module._reset_state()
    spike_selector_module._init_state(4)
    s_in = torch.ones((4, 10), device=device)
    currents, n_currents = spike_selector_module(s_in, is_input=True)
    assert currents.sum() == 80
    assert n_currents.sum() == 80

def test_prioritized_input_forward():
    device = torch.device("cpu")
    forward_module = SparseFullyLinearArchitecture(n_inputs=2, neurons_per_layer=[3,4], device=device)
    weights_per_layer = [
        torch.ones((3, 2), dtype=torch.float32),
        torch.ones((4, 3), dtype=torch.float32),
    ]
    state_dict = SparseFullyLinearArchitecture.create_state_dict_from_weights_list(weights_per_layer)
    forward_module.load_state_dict(state_dict)
    spike_scheduler = RandomSpikeScheduler(None)
    spike_selector_module = SpikeSelector(forward_module, spike_scheduler, 2, device, prioritize_input=True)
    spike_selector_module._init_state(2)
    s_in = torch.ones((2, 2), device=device)
    currents, n_currents = spike_selector_module(s_in, is_input=True)
    assert currents.sum() == 12
    assert n_currents.sum() == 12
    assert spike_selector_module.outgoing_spikes.sum() == 0

def test_non_prioritized_input_forward():
    device = torch.device("cpu")
    forward_module = SparseFullyLinearArchitecture(n_inputs=3, neurons_per_layer=[3,4], device=device)
    weights_per_layer = [
        torch.ones((3, 3), dtype=torch.float32),
        torch.ones((4, 3), dtype=torch.float32),
    ]
    state_dict = SparseFullyLinearArchitecture.create_state_dict_from_weights_list(weights_per_layer)
    forward_module.load_state_dict(state_dict)
    spike_scheduler = RandomSpikeScheduler(None)
    spike_selector_module = SpikeSelector(forward_module, spike_scheduler, 2, device, prioritize_input=False)
    spike_selector_module._init_state(2)
    s_in = torch.ones((2, 3), device=device)
    currents, n_currents = spike_selector_module(s_in, is_input=True)
    assert currents.sum() == 12
    assert n_currents.sum() == 12
    assert spike_selector_module.outgoing_spikes.sum() == 2
    assert spike_selector_module.outgoing_spikes[0].sum() == 1

def test_neuron_activity_forward():
    device = torch.device("cpu")
    forward_module = SparseFullyLinearArchitecture(n_inputs=3, neurons_per_layer=[3,3], device=device)
    weights_per_layer = [
        torch.ones((3, 3), dtype=torch.float32),
        torch.ones((3, 3), dtype=torch.float32),
    ]
    state_dict = SparseFullyLinearArchitecture.create_state_dict_from_weights_list(weights_per_layer)
    forward_module.load_state_dict(state_dict)
    spike_scheduler = RandomSpikeScheduler(None)
    spike_selector_module = SpikeSelector(forward_module, spike_scheduler, 2, device, prioritize_input=False)
    spike_selector_module._init_state(2)
    s_in = torch.asarray([[1,1,1,0,0,0],[1,1,1,0,0,0]], dtype=torch.float32, device=device)
    currents, n_currents = spike_selector_module(s_in, is_input=False)
    assert currents.sum() == 12
    assert n_currents.sum() == 12
    assert spike_selector_module.outgoing_spikes.sum() == 2
    assert spike_selector_module.outgoing_spikes[0].sum() == 1
    assert not spike_selector_module.is_done()

def test_neuron_activity_forward_prioritized_input():
    device = torch.device("cpu")
    forward_module = SparseFullyLinearArchitecture(n_inputs=3, neurons_per_layer=[3,3], device=device)
    weights_per_layer = [
        torch.ones((3, 3), dtype=torch.float32),
        torch.ones((3, 3), dtype=torch.float32),
    ]
    state_dict = SparseFullyLinearArchitecture.create_state_dict_from_weights_list(weights_per_layer)
    forward_module.load_state_dict(state_dict)
    spike_scheduler = RandomSpikeScheduler(None)
    spike_selector_module = SpikeSelector(forward_module, spike_scheduler, 2, device, prioritize_input=True)
    spike_selector_module._init_state(2)
    s_in = torch.asarray([[1,1,1,0,0,0],[1,1,1,0,0,0]], dtype=torch.float32, device=device)
    currents, n_currents = spike_selector_module(s_in, is_input=False)
    assert currents.sum() == 12
    assert n_currents.sum() == 12
    assert spike_selector_module.outgoing_spikes.sum() == 2
    assert spike_selector_module.outgoing_spikes[0].sum() == 1
    assert not spike_selector_module.is_done()