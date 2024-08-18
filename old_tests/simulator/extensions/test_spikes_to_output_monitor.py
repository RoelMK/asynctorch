from asynctorch.simulator.extensions.spikes_to_output_monitor import (
    SpikesToOutputMonitor,
)
import torch

from old_tests.simulator.test_async_simulator import build_async_simulator


def test_spikes_to_output_monitor_single_step():
    monitor = SpikesToOutputMonitor(4, 2)
    async_simulator = build_async_simulator(
        n_inputs=2, neurons_per_layer=[2, 2], forward_step_extensions=[monitor]
    )
    async_simulator.init_state(batch_size=2)
    s_in = torch.ones((2, 2))
    spike_counts = async_simulator(s_in, dt=1.0)
    assert len(monitor.step_classifications) == 1
    assert torch.allclose(spike_counts[:, 2:], monitor.step_classifications[0][1])
    monitor._reset_state()
    s_in = torch.zeros((2, 2))
    spike_counts = async_simulator(s_in, dt=1.0)
    assert len(monitor.step_classifications) == 0
    monitor._reset_state()
    s_in = torch.ones((2, 2))
    spike_counts = async_simulator(s_in, dt=1.0)
    assert len(monitor.step_classifications) == 1
    assert torch.allclose(spike_counts[:, 2:], monitor.step_classifications[0][1])

def test_spikes_to_output_monitor_multi_step():
    monitor = SpikesToOutputMonitor(1, 1)
    async_simulator = build_async_simulator(
        n_inputs=3,
        forward_group_size=1,
        membrane_threshold=0.5,
        weight_value=0.9,
        neurons_per_layer=[1],
        forward_step_extensions=[monitor],
        apply_refrac=False,
        prioritize_input=False,
    )
    async_simulator.init_state(batch_size=2)
    s_in = torch.ones((2, 3))
    spike_counts = async_simulator(s_in, dt=1.0)
    assert monitor.step_classifications[-1][0][0] == 3
    assert torch.allclose(spike_counts[:, 1:], monitor.step_classifications[-1][1])


def test_spikes_to_output_monitor_compute_latencies():
    monitor = SpikesToOutputMonitor(3, 1)
    async_simulator = build_async_simulator(
        n_inputs=1,
        forward_group_size=1,
        membrane_threshold=0.5,
        weight_value=0.9,
        neurons_per_layer=[1,1,1],
        forward_step_extensions=[monitor],
        apply_refrac=False,
        prioritize_input=False,
    )
    async_simulator.init_state(batch_size=2)
    # Ones
    s_in = torch.ones((2, 1))
    async_simulator(s_in, dt=1.0)
    targets = torch.tensor([0, 0])
    n_spikes_to_any_classification, n_spikes_to_correct_classification = monitor.compute_latencies(targets)
    assert n_spikes_to_any_classification == [3,3]
    assert n_spikes_to_correct_classification == [3,3]
    # Zeros
    monitor._reset_state()
    s_in = torch.zeros((2, 1))
    async_simulator(s_in, dt=1.0)
    targets = torch.tensor([0, 0])
    n_spikes_to_any_classification, n_spikes_to_correct_classification = monitor.compute_latencies(targets)
    assert n_spikes_to_any_classification == []
    assert n_spikes_to_correct_classification == []