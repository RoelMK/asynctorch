from asynctorch.simulator.extensions.currents_to_spike_monitor import CurrentsToSpikeMonitor
import torch
from old_tests.simulator.test_async_simulator import build_async_simulator

def test_currents_to_spike_monitor_isolated():
    monitor = CurrentsToSpikeMonitor(3,0, torch.device('cpu'))
    monitor._init_state(2)
    new_currents = torch.tensor([[1, -1, 2], [2, -2, 3]], dtype=torch.float32)
    n_new_currents = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float32)
    monitor.monitor_current(
        new_currents=new_currents,
        n_new_currents=n_new_currents,
        is_input=False,
    )
    assert torch.allclose(monitor.n_currents_to_spike, n_new_currents)
    assert torch.allclose(monitor.sum_excitatory_currents_to_spike, new_currents * (new_currents > 0))
    assert torch.allclose(monitor.sum_inhibitory_currents_to_spike, new_currents * (new_currents < 0))
    monitor.monitor_current(
        new_currents=new_currents,
        n_new_currents=n_new_currents,
        is_input=False,
    )
    assert torch.allclose(monitor.n_currents_to_spike, n_new_currents * 2)
    assert torch.allclose(monitor.sum_excitatory_currents_to_spike, new_currents * (new_currents > 0) * 2)
    assert torch.allclose(monitor.sum_inhibitory_currents_to_spike, new_currents * (new_currents < 0) * 2)
    assert torch.allclose(monitor.spiked, torch.zeros(2, 3, dtype=torch.bool))
    monitor.monitor_spike(
        new_spikes_current_step=torch.tensor([[0, 0, 1], [0, 1, 0]], dtype=torch.bool),
        all_spikes_current_forward=None,
        all_spikes_previous_forwards=None,
        is_input=False,
    )
    assert torch.allclose(monitor.spiked, torch.tensor([[0, 0, 1], [0, 1, 0]], dtype=torch.bool))
    new_currents2 = torch.ones((2, 3), dtype=torch.float32)
    n_new_currents2 = torch.ones((2, 3), dtype=torch.float32)
    monitor.monitor_current(
        new_currents=new_currents2,
        n_new_currents=n_new_currents2,
        is_input=False,
    )
    assert torch.allclose(monitor.n_currents_to_spike, n_new_currents * 2 + torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.float32))
    assert torch.allclose(monitor.sum_excitatory_currents_to_spike, new_currents * (new_currents > 0) * 2 + torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.float32))
    assert torch.allclose(monitor.sum_inhibitory_currents_to_spike, new_currents * (new_currents < 0) * 2)
    monitor.log()
    assert monitor.n_currents_to_spike_log == [2.0, 4.0]
    assert monitor.sum_excitatory_currents_to_spike_log == [4, 0]
    assert monitor.sum_inhibitory_currents_to_spike_log == [0, -4]
    monitor._init_state(2)
    monitor.log()
    assert torch.allclose(monitor.n_currents_to_spike, torch.zeros(2, 3, dtype=torch.float32))
    assert torch.allclose(monitor.sum_excitatory_currents_to_spike, torch.zeros(2, 3, dtype=torch.float32))
    assert torch.allclose(monitor.sum_inhibitory_currents_to_spike, torch.zeros(2, 3, dtype=torch.float32))
    assert torch.allclose(monitor.spiked, torch.zeros(2, 3, dtype=torch.bool))

def test_currents_to_spike_monitor_isolated_ignore_first_n_neurons():
    monitor = CurrentsToSpikeMonitor(3,1, torch.device('cpu'))
    monitor._init_state(2)
    new_currents = torch.tensor([[1, -1, 2], [2, -2, 3]], dtype=torch.float32)
    n_new_currents = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float32)
    monitor.monitor_current(
        new_currents=new_currents,
        n_new_currents=n_new_currents,
        is_input=False,
    )
    assert torch.allclose(monitor.n_currents_to_spike, n_new_currents[:, 1:])
    assert torch.allclose(monitor.sum_excitatory_currents_to_spike, new_currents[:, 1:] * (new_currents[:, 1:] > 0))
    assert torch.allclose(monitor.sum_inhibitory_currents_to_spike, new_currents[:, 1:] * (new_currents[:, 1:] < 0))

def test_currents_to_spike_monitor_simulated():
    monitor = CurrentsToSpikeMonitor(2, 1, torch.device('cpu'))
    async_simulator = build_async_simulator(
        n_inputs=1,
        forward_group_size=1,
        membrane_threshold=0.5,
        weight_value=0.4,
        neurons_per_layer=[1, 1],
        forward_step_extensions=[monitor],
        apply_refrac=False,
        prioritize_input=False,
        tau_m=1e6,
    )
    async_simulator.init_state(batch_size=2)
    s_in = torch.ones((2, 1))
    async_simulator(s_in, dt=1.0)
    assert monitor.n_currents_to_spike_log == []
    s_in = torch.zeros((2, 1))
    async_simulator(s_in, dt=1.0)
    assert monitor.n_currents_to_spike_log == []
    s_in = torch.ones((2, 1))
    async_simulator(s_in, dt=1.0)
    assert monitor.n_currents_to_spike_log == []
    s_in = torch.ones((2, 1))
    async_simulator(s_in, dt=1.0)
    assert monitor.n_currents_to_spike_log == []
    s_in = torch.ones((2, 1))
    async_simulator(s_in, dt=1.0)
    assert monitor.n_currents_to_spike_log == [1.0, 1.0]
    assert torch.allclose(torch.as_tensor(monitor.sum_excitatory_currents_to_spike_log), torch.tensor([0.4, 0.4], dtype=torch.float32))
    assert torch.allclose(torch.as_tensor(monitor.sum_inhibitory_currents_to_spike_log), torch.tensor([0.0, 0.0], dtype=torch.float32))
