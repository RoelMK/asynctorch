from typing import Tuple
from torch import nn
from abc import ABCMeta, abstractmethod
import torch

class BaseArchitecture(nn.Module, metaclass=ABCMeta):
    def __init__(self, n_inputs, n_neurons, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.device = device

    @abstractmethod
    def forward(self, s: torch.Tensor, is_input: bool, is_input_and_neurons_combined: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes currents from spikes.

        Args:
            s: shape = (n_neurons) or (n_inputs) or (n_neurons+n_inputs) - spikes
            is_input: Whether the spikes are input spikes, and shape is (n_inputs) or (n_neurons), ignore shape if is_input_and_neurons_combined is True
            is_input_and_neurons_combined: Whether the spikes also include input spikes, and shape is (n_neurons+n_inputs)
        Returns:
            I: shape = (n_neurons) - currents
            n_I: shape = (n_neurons) - number of currents
        """
        pass

