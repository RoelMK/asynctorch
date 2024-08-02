import torch
from typing import Any, List, Mapping, Tuple
from torch import nn

from asynctorch.nn.architecture.base_architecture import BaseArchitecture


class SparseFullyLinearArchitecture(BaseArchitecture):
    weights: nn.Parameter # shape = (n_neurons, n_neurons+n_inputs)
    weights_mask: torch.Tensor

    def __init__(self, n_inputs: int, neurons_per_layer: List[int], device: torch.device, *args, **kwargs):
        self.neurons_per_layer = neurons_per_layer
        super().__init__(n_inputs, sum(neurons_per_layer), device, *args, **kwargs)
        self.init_state()

    def forward(self, s: torch.Tensor, is_input: bool, is_input_and_neurons_combined: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        # Make sure we always have shape (n_neurons+n_inputs)
        if not is_input_and_neurons_combined:
            if is_input:
                s = torch.cat((s, torch.zeros((s.shape[0], self.n_neurons), device=self.device)), dim=1)
            else:
                s = torch.cat((torch.zeros((s.shape[0], self.n_inputs), device=self.device), s), dim=1)
        currents = torch.matmul(self.weights, s.T).T
        n_currents = torch.matmul(self.weights.abs().clip(0, 1).ceil(), s.T).T
        return currents, n_currents
    
    def init_state(self):
        weights_list = []
        for i in range(len(self.neurons_per_layer)):
            if i == 0:
                n_neurons_in = self.n_inputs
            else:
                n_neurons_in = self.neurons_per_layer[i-1]
            n_neurons_out = self.neurons_per_layer[i]
            weights_list.append(
                torch.zeros(
                    (n_neurons_out, n_neurons_in),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            torch.nn.init.xavier_uniform_(weights_list[-1])
        state_dict = SparseFullyLinearArchitecture.create_state_dict_from_weights_list(weights_list)
        self.load_state_dict(state_dict)
    
    def load_state_dict(self, state_dict: Mapping[str, Any], *args, strict: bool = False, **kwargs):
        weights: torch.Tensor = state_dict["weights"]
        if len(weights.shape) != 2:
            raise ValueError("Weights must be 2-dimensional")
        if weights.shape[1] < weights.shape[0]:
            raise ValueError("Weights must be of shape (n_neurons, n_neurons+n_inputs), but is transposed or invalid")
        self.weights = nn.Parameter(weights)
        self.register_buffer("weights_mask", state_dict["weights_mask"])
        self.weights.register_hook(lambda grad: grad * self.weights_mask)

    def get_weights_list(self) -> List[torch.Tensor]:
        """
        Returns:
            List[torch.Tensor]: List of weights for each layer.
        """
        weights = []
        n_neuron_layers = len(self.neurons_per_layer)
        for i in range(n_neuron_layers):
            neurons_in_start = sum(self.neurons_per_layer[0 : i])+self.n_inputs
            neurons_in_end = sum(self.neurons_per_layer[0 : i+1])+self.n_inputs
            neurons_out_start = sum(self.neurons_per_layer[0 : i+1])
            neurons_out_end = sum(self.neurons_per_layer[0 : i + 2])

            weights.append(self.weights[
                neurons_out_start : neurons_out_end,
                neurons_in_start : neurons_in_end,
            ])
        return weights

    def create_state_dict_from_weights_list(weights_list: List[torch.Tensor]) -> Mapping[str, Any]:
        """
        Builds the weights matrix from a list of tensors.

        Args:
            weights_list: A list of tensors, where each tensor represents a layer transformation.
        Returns:
            weights: shape = (n_neurons, n_neurons+n_inputs)
            weights_mask: shape = (n_neurons, n_neurons+n_inputs) - 1 where weights are trainable, 0 where weights are fixed
        """
        # 1. Acquire shapes
        # >> Input layer
        n_inputs = weights_list[0].shape[1]
        # >> Hidden layer
        n_neuron_layers = len(weights_list) - 1
        n_neurons_per_layer = [weights_list[i].shape[0] for i in range(0, len(weights_list))]
        n_neurons = sum(n_neurons_per_layer)

        # 2. Build weights matrix and mask
        weights = torch.zeros((n_neurons, n_neurons+n_inputs), dtype=torch.float32, device=weights_list[0].device)
        weights_mask = torch.zeros((n_neurons, n_neurons+n_inputs), dtype=torch.bool, device=weights_list[0].device)
        # >> Input layer
        weights[0:n_neurons_per_layer[0], 0:n_inputs] = weights_list[0]
        weights_mask[0:n_neurons_per_layer[0], 0:n_inputs] = True
        # >> Hidden layers and output layer
        for i in range(n_neuron_layers):
            neurons_in_start = sum(n_neurons_per_layer[0 : i])+n_inputs
            neurons_in_end = sum(n_neurons_per_layer[0 : i+1])+n_inputs
            neurons_out_start = sum(n_neurons_per_layer[0 : i+1])
            neurons_out_end = sum(n_neurons_per_layer[0 : i + 2])

            weights[
                neurons_out_start : neurons_out_end,
                neurons_in_start : neurons_in_end,
            ] = weights_list[i+1]
            weights_mask[
                neurons_out_start : neurons_out_end,
                neurons_in_start : neurons_in_end,
            ] = True
        return {"weights": weights, "weights_mask": weights_mask}