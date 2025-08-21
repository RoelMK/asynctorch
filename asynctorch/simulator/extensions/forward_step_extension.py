from typing import Tuple
import torch


class ForwardStepExtension:
    def __init__(self):
        pass

    def on_spike(
        self,
        new_spikes_current_step: torch.Tensor,
        all_spikes_current_forward: torch.Tensor,
        all_spikes_previous_forwards: torch.Tensor,
        is_input: bool,
    ) -> torch.Tensor:
        return new_spikes_current_step

    def on_current(
        self, new_currents: torch.Tensor, n_new_currents: torch.Tensor, is_input: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return new_currents, n_new_currents

    def force_stop(self) -> bool:
        return False
    
    def on_done(self, final_spike_counts: torch.Tensor):
        pass

    def _init_state(self, batch_size: int):
        pass

    def _reset_state(self):
        pass

    def _detach_state(self):
        pass


class ForwardStepMonitor(ForwardStepExtension):
    def __init__(self):
        super().__init__()

    def on_spike(self, new_spikes_current_step, all_spikes_current_forward, all_spikes_previous_forwards, is_input):
        self.monitor_spike(new_spikes_current_step, all_spikes_current_forward, all_spikes_previous_forwards, is_input)
        return new_spikes_current_step

    def on_current(self, new_currents, n_new_currents, is_input):
        self.monitor_current(new_currents, n_new_currents, is_input)
        return new_currents, n_new_currents

    def monitor_spike(
        self,
        new_spikes_current_step: torch.Tensor,
        all_spikes_current_forward: torch.Tensor,
        all_spikes_previous_forwards: torch.Tensor,
        is_input: bool,
    ) -> None:
        pass

    def monitor_current(self, new_currents: torch.Tensor, n_new_currents: torch.Tensor, is_input: bool) -> None:
        pass
