import torch
import torch.nn as nn
from asynctorch.nn.architecture.base_architecture import BaseArchitecture
from asynctorch.simulator.spike_scheduler import SpikeScheduler
from typing import Tuple


class SpikeSelector(nn.Module):
    # State variables
    # If prioritize_input is False, then shape is (batch_size, n_neurons) instead
    outgoing_spikes: torch.Tensor  # shape = (batch_size, (n_inputs+)n_neurons)
    indices: torch.Tensor  # shape = (batch_size, (n_inputs+)n_neurons)
    indices_by_priority: torch.Tensor  # shape = (batch_size, (n_inputs+)n_neurons)

    def __init__(
        self,
        architecture: BaseArchitecture,
        spike_scheduler: SpikeScheduler,
        forward_group_size: int,
        device: torch.device,
        prioritize_input: bool
    ):
        super().__init__()
        self.architecture = architecture
        self.spike_scheduler = spike_scheduler
        self.prioritize_input = prioritize_input
        self.device = device
        self.forward_group_size = forward_group_size
        self._reset_state()

    def is_init(self) -> bool:
        return self.outgoing_spikes is not None

    def is_done(self) -> bool:
        return self.is_init() and self.outgoing_spikes.sum() == 0

    def _init_state(self, batch_size: int):
        if self.is_init():
            raise RuntimeError("Cannot initialize state twice")
        if self.prioritize_input:
            n_indices = self.architecture.n_neurons
        else:
            n_indices = self.architecture.n_inputs + self.architecture.n_neurons
        self.outgoing_spikes = torch.zeros((batch_size, n_indices), device=self.device, dtype=torch.float)
        self.indices = torch.arange(n_indices, dtype=torch.int64, device=self.device).repeat(batch_size, 1)
        self.indices_by_priority = self.indices.clone()

    def _reset_state(self):
        self.outgoing_spikes = None
        self.indices = None
        self.indices_by_priority = None

    def _detach_state(self):
        if self.is_init():
            self.outgoing_spikes = self.outgoing_spikes.detach()
            self.indices = self.indices.detach()
            self.indices_by_priority = self.indices_by_priority.detach()

    def sort_indices(self):
        self.indices_by_priority = self.spike_scheduler.sort(self.outgoing_spikes, self.indices)

    def emit_spikes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Select spikes to emit
        emit_spike_indices = self.indices_by_priority[:, : self.forward_group_size]
        emit_spike_mask = torch.zeros_like(self.indices, dtype=torch.bool)
        emit_spike_mask = emit_spike_mask.scatter(1, emit_spike_indices, True)

        # Select spikes and then convert them into currents
        outgoing_spikes_masked = self.outgoing_spikes * emit_spike_mask
        self.outgoing_spikes = (self.outgoing_spikes * ~emit_spike_mask) # - emit_spike_mask.int()).clamp(min=0)
        return self.architecture(outgoing_spikes_masked, is_input=False, is_input_and_neurons_combined=not self.prioritize_input)

    def forward(self, s_in: torch.Tensor, is_input: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes the input spikes, then computes the new currents.

        Args:
            s_in: shape = (n_inputs) if is_input else (n_neurons)
            is_input: whether the input spikes are from the input layer or not.
                    If prioritizing input, then this will be used to fast-forward the input spikes.
        Returns:
            I_new: shape = (n_neurons) - value of new currents per neuron
            n_I_new: shape = (n_neurons) - number of new currents per neuron
        """
        if not self.is_init():
            raise RuntimeError("Must initialize state before calling forward")
        # If prioritizing input (and if it is input), then we can fast-forward the spikes.
        if is_input and self.prioritize_input:
            return self.architecture(s_in, is_input=True, is_input_and_neurons_combined=False)
        # Otherwise, we need to add the spikes to the outgoing spikes to be processed in groups.
        if not self.prioritize_input:
            # Make sure the spikes are in the right shape, since we also need to track input spikes in outgoing_spikes.
            if is_input:
                print(s_in.shape)
                s_in = torch.cat((s_in, torch.zeros((s_in.shape[0], self.architecture.n_neurons), device=self.device)), dim=1)
            else:
                s_in = torch.cat((torch.zeros((s_in.shape[0], self.architecture.n_inputs), device=self.device), s_in), dim=1)
        self.outgoing_spikes = self.outgoing_spikes + s_in
        self.sort_indices()
        return self.emit_spikes()
