from asynctorch.simulator.extensions.stop_on_output_extension import StopOnOutputExtension
import torch

from tests.simulator.test_async_simulator import build_async_simulator


def test_stop_on_output_without_post_output():
    extension = StopOnOutputExtension(3, 1, 0)
    new_spikes = torch.zeros(2, 3, dtype=torch.float32)
    new_spikes_out = extension.on_spike(
        new_spikes_current_step=new_spikes,
        all_spikes_current_forward=None,
        all_spikes_previous_forwards=None,
        is_input=True,
    )
    assert torch.allclose(new_spikes, new_spikes_out)
    assert extension.outputs_seen.shape == (2,)
    assert extension.n_post_output_steps.shape == (2,)
    new_spikes = torch.ones(2, 3, dtype=torch.float32)
    new_spikes[:, 2] = 0
    new_spikes_out = extension.on_spike(
        new_spikes_current_step=new_spikes,
        all_spikes_current_forward=None,
        all_spikes_previous_forwards=None,
        is_input=False,
    )
    assert torch.allclose(new_spikes, new_spikes_out)
    new_spikes = torch.ones(2, 3, dtype=torch.float32)
    new_spikes[:, 2] = 0
    new_spikes_out = extension.on_spike(
        new_spikes_current_step=new_spikes,
        all_spikes_current_forward=None,
        all_spikes_previous_forwards=None,
        is_input=False,
    )
    assert torch.allclose(new_spikes, new_spikes_out)
    new_spikes = torch.zeros(2, 3, dtype=torch.float32)
    new_spikes[1, 2] = 1
    new_spikes_out = extension.on_spike(
        new_spikes_current_step=new_spikes,
        all_spikes_current_forward=None,
        all_spikes_previous_forwards=None,
        is_input=False,
    )
    assert torch.allclose(new_spikes, new_spikes_out)
    assert torch.allclose(extension.outputs_seen, torch.tensor([False, True]))
    assert not extension.force_stop()
    new_spikes = torch.ones(2, 3, dtype=torch.float32)
    new_spikes_out = extension.on_spike(
        new_spikes_current_step=new_spikes,
        all_spikes_current_forward=None,
        all_spikes_previous_forwards=None,
        is_input=False,
    )
    assert torch.allclose(new_spikes_out, torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.float32))
    assert torch.allclose(extension.outputs_seen, torch.tensor([True, True]))
    assert not extension.force_stop()
    new_spikes = torch.ones(2, 3, dtype=torch.float32)
    new_spikes_out = extension.on_spike(
        new_spikes_current_step=new_spikes,
        all_spikes_current_forward=None,
        all_spikes_previous_forwards=None,
        is_input=False,
    )
    assert torch.allclose(new_spikes_out, torch.zeros((2,3), dtype=torch.float32))
    assert torch.allclose(extension.outputs_seen, torch.tensor([True, True]))
    assert extension.force_stop()


def test_stop_on_output_with_post_output():
    extension = StopOnOutputExtension(3, 1, 2)
    new_spikes = torch.zeros(2, 3, dtype=torch.float32)
    new_spikes[0, 2] = 1
    new_spikes_out = extension.on_spike(
        new_spikes_current_step=new_spikes,
        all_spikes_current_forward=None,
        all_spikes_previous_forwards=None,
        is_input=True,
    )
    assert torch.allclose(new_spikes, new_spikes_out)
    assert torch.allclose(extension.outputs_seen, torch.tensor([False, False]))
    assert torch.allclose(extension.n_post_output_steps, torch.tensor([0, 0]))
    new_spikes_out = extension.on_spike(
        new_spikes_current_step=new_spikes,
        all_spikes_current_forward=None,
        all_spikes_previous_forwards=None,
        is_input=False,
    )
    assert torch.allclose(new_spikes, new_spikes_out)
    assert torch.allclose(extension.outputs_seen, torch.tensor([True, False]))
    assert torch.allclose(extension.n_post_output_steps, torch.tensor([1, 0])) 
    extension.on_spike(
        new_spikes_current_step=torch.zeros(2, 3, dtype=torch.float32),
        all_spikes_current_forward=None,
        all_spikes_previous_forwards=None,
        is_input=False,
    )
    assert torch.allclose(extension.outputs_seen, torch.tensor([True, False]))
    assert torch.allclose(extension.n_post_output_steps, torch.tensor([2, 0]))
    new_spikes_out = extension.on_spike(
        new_spikes_current_step=new_spikes,
        all_spikes_current_forward=None,
        all_spikes_previous_forwards=None,
        is_input=False,
    )
    assert torch.allclose(new_spikes_out, new_spikes)
    assert torch.allclose(extension.outputs_seen, torch.tensor([True, False]))
    assert torch.allclose(extension.n_post_output_steps, torch.tensor([3, 0]))
    assert not extension.force_stop()
    new_spikes_out = extension.on_spike(
        new_spikes_current_step=new_spikes,
        all_spikes_current_forward=None,
        all_spikes_previous_forwards=None,
        is_input=False,
    )
    assert torch.allclose(extension.n_post_output_steps, torch.tensor([4, 0]))
    assert torch.allclose(new_spikes_out, torch.zeros(2, 3, dtype=torch.float32))
    new_spikes = torch.ones(2, 3, dtype=torch.float32)
    new_spikes_out = extension.on_spike(
        new_spikes_current_step=new_spikes,
        all_spikes_current_forward=None,
        all_spikes_previous_forwards=None,
        is_input=False,
    )
    assert torch.allclose(new_spikes_out, torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32))
    assert torch.allclose(extension.outputs_seen, torch.tensor([True, True]))
    assert torch.allclose(extension.n_post_output_steps, torch.tensor([5, 1]))
    assert not extension.force_stop()
    new_spikes_out = extension.on_spike(
        new_spikes_current_step=new_spikes,
        all_spikes_current_forward=None,
        all_spikes_previous_forwards=None,
        is_input=False,
    )
    assert torch.allclose(extension.n_post_output_steps, torch.tensor([6, 2]))
    assert not extension.force_stop()
    new_spikes_out = extension.on_spike(
        new_spikes_current_step=new_spikes,
        all_spikes_current_forward=None,
        all_spikes_previous_forwards=None,
        is_input=False,
    )
    assert torch.allclose(extension.n_post_output_steps, torch.tensor([7, 3]))
    assert not extension.force_stop()
    new_spikes_out = extension.on_spike(
        new_spikes_current_step=new_spikes,
        all_spikes_current_forward=None,
        all_spikes_previous_forwards=None,
        is_input=False,
    )
    assert torch.allclose(extension.n_post_output_steps, torch.tensor([8, 4]))
    assert torch.allclose(new_spikes_out, torch.zeros(2, 3, dtype=torch.float32))
    assert extension.force_stop()


def test_stop_on_output_in_async_sim_without_post_output():
    extension = StopOnOutputExtension(1, 1, 0)
    async_simulator = build_async_simulator(n_inputs=1, neurons_per_layer=[1], membrane_threshold=0.5, forward_step_extensions=[extension])
    async_simulator.init_state(batch_size=2)
    s_in = torch.asarray([[1], [0]], dtype=torch.float32)
    spike_counts = async_simulator(s_in, dt=1.0)
    assert torch.allclose(spike_counts, torch.tensor([[1], [0]], dtype=torch.float32))
    assert torch.allclose(extension.outputs_seen, torch.tensor([True, False]))
    assert not extension.force_stop()
    s_in = torch.asarray([[0], [1]], dtype=torch.float32)
    spike_counts = async_simulator(s_in, dt=1.0)
    assert torch.allclose(spike_counts, torch.tensor([[0], [1]], dtype=torch.float32))
    assert torch.allclose(extension.outputs_seen, torch.tensor([False, True]))
    assert not extension.force_stop()
    s_in = torch.asarray([[1], [1]], dtype=torch.float32)
    spike_counts = async_simulator(s_in, dt=1.0)
    assert torch.allclose(spike_counts, torch.tensor([[1], [1]], dtype=torch.float32))
    assert torch.allclose(extension.outputs_seen, torch.tensor([True, True]))
    assert torch.allclose(extension.n_post_output_steps, torch.tensor([1, 1]))
