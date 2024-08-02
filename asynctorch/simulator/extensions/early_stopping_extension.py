from asynctorch.simulator.extensions.forward_step_extension import ForwardStepExtension


class EarlyStoppingExtension(ForwardStepExtension):
    def __init__(self, stop_after_n_steps: int):
        super().__init__()
        self.stop_after_n_steps = stop_after_n_steps
        self.i_step = 0

    def on_spike(self, new_spikes_current_step, all_spikes_current_forward, all_spikes_previous_forwards, is_input):
        if is_input:
            self.i_step = 0
        else:
            self.i_step += 1
        return new_spikes_current_step

    def force_stop(self):
        return self.i_step >= self.stop_after_n_steps