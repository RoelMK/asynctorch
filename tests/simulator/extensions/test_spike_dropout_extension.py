from asynctorch.simulator.extensions.spike_dropout_extension import SpikeDropoutExtension
import torch

def test_dropout_extension():
    extension = SpikeDropoutExtension(p=1.0, apply_to_input=True, apply_to_network=False)
    s_in = torch.ones((2,10), dtype=torch.float32)
    spikes = extension.on_spike(s_in, None, None, is_input=True)
    assert spikes.sum() == 0
    spikes = extension.on_spike(s_in, None, None, is_input=False)
    assert spikes.sum() == 20
    extension = SpikeDropoutExtension(p=1.0, apply_to_input=False, apply_to_network=True)
    spikes = extension.on_spike(s_in, None, None, is_input=True)
    assert spikes.sum() == 20
    spikes = extension.on_spike(s_in, None, None, is_input=False)
    assert spikes.sum() == 0
    extension = SpikeDropoutExtension(p=1.0, apply_to_input=True, apply_to_network=True)
    spikes = extension.on_spike(s_in, None, None, is_input=True)
    assert spikes.sum() == 0
    spikes = extension.on_spike(s_in, None, None, is_input=False)
    assert spikes.sum() == 0
    extension = SpikeDropoutExtension(p=0.0, apply_to_input=True, apply_to_network=True)
    spikes = extension.on_spike(s_in, None, None, is_input=True)
    assert spikes.sum() == 20
    spikes = extension.on_spike(s_in, None, None, is_input=False)
    assert spikes.sum() == 20