import torch
from typing import Any, List, Mapping, Tuple
from torch import nn
import torch.nn.functional as F

from asynctorch.nn.architecture.base_architecture import BaseArchitecture

# filters are of shape (out_channels, in_channels, kernel_height, kernel_width)
# stride and padding are extra parameters
class Conv2dParams:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, is_avg_pool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size # for now, we assume kernel is square
        self.stride = stride
        self.padding = padding
        self.is_avg_pool = is_avg_pool
        if is_avg_pool and in_channels != out_channels:
            raise ValueError("Number of input channels must match number of output channels for average pooling")

class Conv2dArchitecture(BaseArchitecture):
    conv_filters: nn.ParameterList # List of weights (nn.Parameter) for each convolutional filter, each are of shape are of shape (out_channels, in_channels, kernel_height, kernel_width)
    fc_weights: nn.Parameter # Weights for the final fully connected layer, shape (n_neurons output layer, n_neurons last hidden layer)

    def compute_shape_per_layer(self):
        self.input_shape_per_layer = [(self.input_channels, self.input_height, self.input_width)]
        for i in range(len(self.neurons_per_layer) - 1):
            prev_shape = self.input_shape_per_layer[-1]
            conv_params = self.conv_params_per_layer[i]
            out_height = (prev_shape[1] - conv_params.kernel_size + 2 * conv_params.padding) // conv_params.stride + 1
            out_width = (prev_shape[2] - conv_params.kernel_size + 2 * conv_params.padding) // conv_params.stride + 1
            self.input_shape_per_layer.append((conv_params.out_channels, out_height, out_width))

    def __init__(self, input_channels: int, input_height: int, input_width: int, neurons_per_layer: List[int], conv_params_per_layer: List[Conv2dParams], device: torch.device, *args, **kwargs):
        self.neurons_per_layer = neurons_per_layer
        self.conv_params_per_layer = conv_params_per_layer
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        n_inputs = input_channels * input_height * input_width
        if len(neurons_per_layer) != len(conv_params_per_layer) + 1:
            raise ValueError("Number of convolutional layers must be one less than number of neuron layers. Final layer is fully connected.")
        super().__init__(n_inputs, sum(neurons_per_layer), device, *args, **kwargs)
        self.compute_shape_per_layer()
        self.init_state()

    def forward(self, s: torch.Tensor, is_input: bool, is_input_and_neurons_combined: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if is_input_and_neurons_combined:
            raise RuntimeError("is_input_and_neurons_combined is not supported by this architecture. Input must be prioritized.")
        if is_input:
            # Reshape input spikes into 2d image
            s = s.view(-1, self.input_channels, self.input_height, self.input_width)
            # Apply convolutional filters
            currents = F.conv2d(s, self.conv_filters[0], stride=self.conv_params_per_layer[0].stride, padding=self.conv_params_per_layer[0].padding)
            n_currents = F.conv2d(s, self.conv_filters[0].abs().clip(0, 1).ceil(), stride=self.conv_params_per_layer[0].stride, padding=self.conv_params_per_layer[0].padding)
            # Flatten output
            currents = currents.view(s.shape[0], -1)
            n_currents = n_currents.view(s.shape[0], -1)
            if currents.shape[1] != self.neurons_per_layer[0]:
                raise RuntimeError("Number of neurons in first layer does not match number of output channels")
            # Cat zeros to the end of the currents
            zeros = torch.zeros((s.shape[0], self.n_neurons - currents.shape[1]), dtype=torch.float32, device=self.device)
            currents = torch.cat((currents, zeros), dim=1)
            n_currents = torch.cat((n_currents, zeros), dim=1)
        else:
            # Compute currents for all layers
            currents = torch.zeros((s.shape[0], self.n_neurons), dtype=torch.float32, device=self.device)
            n_currents = torch.zeros((s.shape[0], self.n_neurons), dtype=torch.float32, device=self.device)
            for i in range(len(self.neurons_per_layer)-1):
                # if i < len(self.neurons_per_layer) - 1: apply convolutional filter, else apply fully connected layer
                if i < len(self.neurons_per_layer) - 2:
                    conv_params = self.conv_params_per_layer[i+1]
                    conv_filter = self.conv_filters[i+1]
                    shape = self.input_shape_per_layer[i+1]
                    s_conv = s[:, sum(self.neurons_per_layer[0:i]):sum(self.neurons_per_layer[0:i+1])].view(s.shape[0], shape[0], shape[1], shape[2])
                    conv_currents = F.conv2d(s_conv, conv_filter, stride=conv_params.stride, padding=conv_params.padding)
                    conv_n_currents = F.conv2d(s_conv, conv_filter.abs().clip(0, 1).ceil(), stride=conv_params.stride, padding=conv_params.padding)
                    currents[:, sum(self.neurons_per_layer[0:i+1]):sum(self.neurons_per_layer[0:i+2])] = conv_currents.view(s_conv.shape[0], -1)
                    n_currents[:, sum(self.neurons_per_layer[0:i+1]):sum(self.neurons_per_layer[0:i+2])] = conv_n_currents.view(s_conv.shape[0], -1)
                else:
                    s_fc = s[:, sum(self.neurons_per_layer[0:i]):sum(self.neurons_per_layer[0:i+1])]
                    currents[:, sum(self.neurons_per_layer[0:i+1]):sum(self.neurons_per_layer[0:i+2])] = torch.matmul(s_fc, self.fc_weights.T)
                    n_currents[:, sum(self.neurons_per_layer[0:i+1]):sum(self.neurons_per_layer[0:i+2])] = torch.matmul(s_fc, self.fc_weights.abs().clip(0, 1).ceil().T)
        return currents, n_currents
    
    def init_state(self):
        conv_filters = []
        fc_weights = torch.zeros((self.neurons_per_layer[-1], self.neurons_per_layer[-2]), dtype=torch.float32, device=self.device)
        torch.nn.init.xavier_uniform_(fc_weights)
        for i in range(len(self.conv_params_per_layer)):
            conv_params = self.conv_params_per_layer[i]
            conv_filter = torch.zeros((conv_params.out_channels, conv_params.in_channels, conv_params.kernel_size, conv_params.kernel_size), dtype=torch.float32, device=self.device)
            torch.nn.init.xavier_uniform_(conv_filter)
            conv_filters.append(nn.Parameter(conv_filter))
        state_dict = Conv2dArchitecture.create_state_dict(conv_filters, fc_weights)
        self.load_state_dict(state_dict)
    
    # TODO: match nn.Conv2d state_dict
    def load_state_dict(self, state_dict: Mapping[str, Any], *args, strict: bool = False, **kwargs):
        conv_filters: List[torch.Tensor] = state_dict["conv_filters"]
        fc_weights: torch.Tensor = state_dict["fc_weights"]
        if len(conv_filters) != len(self.conv_params_per_layer):
            raise ValueError("Number of convolutional filters does not match number of convolutional layers")
        if fc_weights.shape != (self.neurons_per_layer[-1], self.neurons_per_layer[-2]):
            raise ValueError("Shape of fully connected weights does not match expected shape")
        self.conv_filters = nn.ParameterList(conv_filters)
        self.fc_weights = nn.Parameter(fc_weights)

    def create_state_dict(conv_filters: List[torch.Tensor], fc_weights: torch.Tensor) -> Mapping[str, Any]:
        state_dict = {
            "conv_filters": nn.ParameterList(conv_filters),
            "fc_weights": nn.Parameter(fc_weights),
        }
        return state_dict