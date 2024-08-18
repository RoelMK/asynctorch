import torch
from asynctorch.simulator.extensions.early_stopping_extension import EarlyStoppingExtension


def test_early_stopping():
    extension = EarlyStoppingExtension(stop_after_n_steps=2)
    s_in = torch.ones((2,10), dtype=torch.float32)
    spikes = extension.on_spike(s_in, None, None, is_input=True)
    assert torch.allclose(spikes, s_in)
    assert not extension.force_stop()
    extension.on_spike(s_in, None, None, is_input=False)
    assert not extension.force_stop()
    extension.on_spike(s_in, None, None, is_input=False)
    assert extension.force_stop()
    extension.on_spike(s_in, None, None, is_input=False)
    assert extension.force_stop()
    extension.on_spike(s_in, None, None, is_input=True)
    assert not extension.force_stop()
    extension.on_spike(s_in, None, None, is_input=False)
    assert not extension.force_stop()
    extension.on_spike(s_in, None, None, is_input=False)
    assert extension.force_stop()