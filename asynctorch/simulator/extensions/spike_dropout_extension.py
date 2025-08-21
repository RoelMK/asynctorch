from asynctorch.simulator.extensions.forward_step_extension import ForwardStepExtension
import torch

class SpikeDropoutExtension(ForwardStepExtension):
    def __init__(self, p: float, apply_to_input: bool, apply_to_network: bool):
        super().__init__()
        self.p = p
        self.apply_to_input = apply_to_input
        self.apply_to_network = apply_to_network

    def on_spike(self, new_spikes_current_step, all_spikes_current_forward, all_spikes_previous_forwards, is_input):
        if (is_input and not self.apply_to_input) or (not is_input and not self.apply_to_network):
            return new_spikes_current_step
        mask = torch.empty(new_spikes_current_step.shape, device=new_spikes_current_step.device).bernoulli_(1-self.p)
        return new_spikes_current_step * mask