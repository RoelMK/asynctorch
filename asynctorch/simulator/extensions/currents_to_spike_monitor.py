import torch
from asynctorch.simulator.extensions.forward_step_extension import ForwardStepMonitor


class CurrentsToSpikeMonitor(ForwardStepMonitor):
    def __init__(self, n_neurons, ignore_first_n_neurons, device):
        super().__init__()
        self.n_neurons = n_neurons
        self.ignore_first_n_neurons = ignore_first_n_neurons
        self.device = device
        self.n_currents_to_spike_log = []
        self.sum_excitatory_currents_to_spike_log = []
        self.sum_inhibitory_currents_to_spike_log = []

    def _init_state(self, batch_size: int):
        self.spiked = torch.zeros(batch_size, self.n_neurons-self.ignore_first_n_neurons, dtype=torch.bool, device=self.device)
        self.n_currents_to_spike = torch.zeros(batch_size, self.n_neurons-self.ignore_first_n_neurons, dtype=torch.float32, device=self.device)
        self.sum_excitatory_currents_to_spike = torch.zeros(batch_size, self.n_neurons-self.ignore_first_n_neurons, dtype=torch.float32, device=self.device)
        self.sum_inhibitory_currents_to_spike = torch.zeros(batch_size, self.n_neurons-self.ignore_first_n_neurons, dtype=torch.float32, device=self.device)
        
    def _reset_state(self):
        self.spiked = None
        self.n_currents_to_spike = None
        self.sum_excitatory_currents_to_spike = None
        self.sum_inhibitory_currents_to_spike = None
        self.n_currents_to_spike_log = []
        self.sum_excitatory_currents_to_spike_log = []
        self.sum_inhibitory_currents_to_spike_log = []

    def log(self):
        # copy all non-zero elements to the log
        if self.spiked.sum() > 0:
            self.n_currents_to_spike_log.extend(self.n_currents_to_spike[self.spiked].cpu().flatten().tolist())
            self.sum_excitatory_currents_to_spike_log.extend(self.sum_excitatory_currents_to_spike[self.spiked].cpu().flatten().tolist())
            self.sum_inhibitory_currents_to_spike_log.extend(self.sum_inhibitory_currents_to_spike[self.spiked].cpu().flatten().tolist())
    
    def monitor_spike(self, new_spikes_current_step, all_spikes_current_forward, all_spikes_previous_forwards, is_input):
        if is_input:
            self._init_state(new_spikes_current_step.size(0))
            return
        self.spiked = self.spiked | (new_spikes_current_step[:, self.ignore_first_n_neurons:] > 0)

    def monitor_current(self, new_currents, n_new_currents, is_input):
        if is_input:
            return
        # update the currents to spike for the neurons that not already spiked
        new_currents_to_spike = new_currents[:, self.ignore_first_n_neurons:] * (1 - self.spiked.float())
        n_new_currents_to_spike = n_new_currents[:, self.ignore_first_n_neurons:] * (1 - self.spiked.float())
        self.n_currents_to_spike = self.n_currents_to_spike + n_new_currents_to_spike
        self.sum_excitatory_currents_to_spike = self.sum_excitatory_currents_to_spike + new_currents_to_spike * (new_currents_to_spike > 0)
        self.sum_inhibitory_currents_to_spike = self.sum_inhibitory_currents_to_spike + new_currents_to_spike * (new_currents_to_spike < 0)

    def on_done(self, final_spike_counts):
        self.log()