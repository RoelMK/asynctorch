import torch
from asynctorch.simulator.extensions.forward_step_extension import ForwardStepExtension


class StopOnOutputExtension(ForwardStepExtension):
    def __init__(self, n_neurons: int, n_outputs: int, max_post_output_steps: int):
        super().__init__()
        self.max_post_output_steps = max_post_output_steps
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs

    def on_spike(self, new_spikes_current_step, all_spikes_current_forward, all_spikes_previous_forwards, is_input):
        batch_size = new_spikes_current_step.shape[0]
        if is_input:
            self.outputs_seen = torch.zeros(batch_size, dtype=torch.bool, device=new_spikes_current_step.device)
            self.n_post_output_steps = torch.zeros(batch_size, dtype=torch.int64, device=new_spikes_current_step.device)
        else:
            new_spikes_current_step = new_spikes_current_step * (self.n_post_output_steps <= self.max_post_output_steps).unsqueeze(1).float()
            self.outputs_seen = self.outputs_seen | torch.any(new_spikes_current_step[:, self.n_neurons-self.n_outputs:] > 0, dim=1)
            self.n_post_output_steps = self.n_post_output_steps + self.outputs_seen.int()
        return new_spikes_current_step

    def force_stop(self):
        return torch.all(self.outputs_seen) and torch.all(self.n_post_output_steps > self.max_post_output_steps+1)