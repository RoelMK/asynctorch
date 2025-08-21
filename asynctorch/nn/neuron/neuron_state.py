from typing import List
import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod


class NeuronState(nn.Module, metaclass=ABCMeta):
    # This module can represent different neuron modules

    def __init__(self, neurons_per_layer: List[int], device: torch.device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neurons_per_layer = neurons_per_layer
        self.n_neurons = sum(neurons_per_layer)
        self.n_inputs = neurons_per_layer[0]
        self.n_outputs = neurons_per_layer[-1]
        self.device = device

    @abstractmethod
    def is_init(self) -> bool:
        pass

    # Initialize state variables, should not be called manually
    @abstractmethod
    def _init_state(self, batch_size: int):
        pass

    # Reset state variables, should not be called manually
    @abstractmethod
    def _reset_state(self):
        pass

    # Detach state variables, should not be called manually
    @abstractmethod
    def _detach_state(self):
        pass

    @abstractmethod
    def get_pre_spike_state(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_post_spike_state(self) -> torch.Tensor:
        pass

    @abstractmethod
    def step_dynamics(self, dt: float):
        pass

    @abstractmethod
    def forward(self, I_new: torch.Tensor, n_I_new: torch.Tensor) -> torch.Tensor:
        """
        Takes the new currents and updates the neuron states.

        Args:
            I_new: shape = (n_neurons) - value of new currents per neuron
            n_I_new: shape = (n_neurons) - number of incoming spikes per neuron
        Returns:
            s_out: shape = (n_neurons) - outgoing spikes
        """
        pass

