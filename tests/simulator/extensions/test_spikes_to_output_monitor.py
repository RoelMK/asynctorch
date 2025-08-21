from asynctorch.simulator.extensions.spikes_to_output_monitor import (
    SpikesToOutputMonitor,
)
import torch

from ..test_async_simulator import build_async_simulator


def test_spikes_to_output_monitor_single_step():
    monitor = SpikesToOutputMonitor(4, 2)
    module_per_layer = [torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 2, bias=False)]
    for module in module_per_layer:
        module.weight.data.fill_(1.0)
    shape_per_layer = [(2,), (2,), (2,)]
    async_simulator = build_async_simulator(
        module_per_layer, shape_per_layer, forward_step_extensions=[monitor]
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
    monitor = SpikesToOutputMonitor(3, 1)
    module_per_layer = [torch.nn.Linear(1, 2, bias=False), torch.nn.Linear(2, 1, bias=False)]
    for module in module_per_layer:
        module.weight.data.fill_(0.9)
    shape_per_layer = [(1,), (2,), (1,)]
    async_simulator = build_async_simulator(
        module_per_layer,
        shape_per_layer,
        forward_group_size=1,
        membrane_threshold=0.5,
        forward_step_extensions=[monitor]
    )
    async_simulator.init_state(batch_size=2)
    s_in = torch.ones((2, 1))
    spike_counts = async_simulator(s_in, dt=1.0)
    assert monitor.step_classifications[-1][0][0] == 3
    assert torch.allclose(spike_counts[:, 1:], monitor.step_classifications[-1][1])

def test_spikes_to_output_monitor_compute_latencies():
    monitor = SpikesToOutputMonitor(3, 1)
    module_per_layer = [torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False)]
    for module in module_per_layer:
        module.weight.data.fill_(0.9)
    shape_per_layer = [(1,), (1,), (1,), (1,)]
    async_simulator = build_async_simulator(
        module_per_layer,
        shape_per_layer,
        forward_group_size=1,
        membrane_threshold=0.5,
        forward_step_extensions=[monitor],
        apply_refrac=False
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