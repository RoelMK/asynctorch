import torch
import torch.nn as nn
import torch.nn.functional as F
from asynctorch.nn.architecture.conv2d_architecture import Conv2dArchitecture, Conv2dParams

def test_init_module():
    device = torch.device("cpu")
    input_channels = 2
    input_height = 2
    input_width = 2
    neurons_per_layer = [12, 8, 2]
    n_neurons = sum(neurons_per_layer)
    conv_params_per_layer = [
        Conv2dParams(in_channels=2, out_channels=3, kernel_size=1, stride=1, padding=0),
        Conv2dParams(in_channels=3, out_channels=2, kernel_size=1, stride=1, padding=0),
    ]
    # (2,2,2) -> (3,2,2) -> (2,2,2) -> (2,)
    module = Conv2dArchitecture(input_channels, input_height, input_width, neurons_per_layer, conv_params_per_layer, device)
    assert len(module.conv_filters) == 2
    assert module.conv_filters[0].shape == (3, 2, 1, 1)
    assert module.conv_filters[1].shape == (2, 3, 1, 1)
    assert module.conv_filters[1].sum() != 0
    assert module.fc_weights.shape == (2, 8)
    assert module.fc_weights.sum() != 0
    
    inp = torch.randn(1, 2, 2, 2).view(1, -1)
    currents, n_currents = module(inp, is_input=True, is_input_and_neurons_combined=False)
    assert currents.shape == (1, n_neurons)
    assert n_currents.shape == (1, n_neurons)

    inp = torch.randn(1, n_neurons)
    currents, n_currents = module(inp, is_input=False, is_input_and_neurons_combined=False)
    assert currents.shape == (1, n_neurons)
    assert n_currents.shape == (1, n_neurons)

def test_simple_conv_correctness():
    device = torch.device("cpu")
    input_channels = 2
    input_height = 2
    input_width = 2
    neurons_per_layer = [12, 8, 2]
    n_neurons = sum(neurons_per_layer)
    conv_params_per_layer = [
        Conv2dParams(in_channels=2, out_channels=3, kernel_size=1, stride=1, padding=0),
        Conv2dParams(in_channels=3, out_channels=2, kernel_size=1, stride=1, padding=0),
    ]
    module = Conv2dArchitecture(input_channels, input_height, input_width, neurons_per_layer, conv_params_per_layer, device)
    
    # Input layer
    inp = torch.ones((1, 2, 2, 2), dtype=torch.float32).view(1, -1)
    currents, n_currents = module(inp, is_input=True, is_input_and_neurons_combined=False)
    s = inp.view(-1, input_channels, input_height, input_width)
    test_currents = F.conv2d(s, module.conv_filters[0], stride=module.conv_params_per_layer[0].stride, padding=module.conv_params_per_layer[0].padding)
    assert torch.allclose(currents[:, :12], test_currents.view(1,-1))
    assert currents[:, 12:].sum() == 0
    assert n_currents[:, 0] == 2
    assert n_currents[:, 12:].sum() == 0

    # Hidden / Output layer
    inp = torch.ones((1, n_neurons), dtype=torch.float32)
    currents, n_currents = module(inp, is_input=False, is_input_and_neurons_combined=False)
    s_conv = inp[:, 0:12].view(-1, 3, 2, 2)
    s_fc = inp[:, 12:20]
    test_conv_currents = F.conv2d(s_conv, module.conv_filters[1], stride=module.conv_params_per_layer[1].stride, padding=module.conv_params_per_layer[1].padding).view(1, -1)
    test_fc_currents = torch.matmul(s_fc, module.fc_weights.T)
    assert currents[:, :12].sum() == 0
    assert torch.allclose(currents[:, 12:20], test_conv_currents)
    assert torch.allclose(currents[:, 20:], test_fc_currents)
    assert n_currents[:, 0:12].sum() == 0
    assert n_currents[0, 12] == 3
    assert n_currents[0, 20] == 8


def test_advanced_conv_correctness():
    # same test, but with Kernel size > 1, stride > 1, padding > 0
    device = torch.device("cpu")
    input_channels = 2
    input_height = 128
    input_width = 128
    neurons_per_layer = [12288, 2048, 2]
    n_neurons = sum(neurons_per_layer)
    conv_params_per_layer = [
        Conv2dParams(in_channels=2, out_channels=3, kernel_size=3, stride=2, padding=1),
        Conv2dParams(in_channels=3, out_channels=2, kernel_size=3, stride=2, padding=1),
    ]
    module = Conv2dArchitecture(input_channels, input_height, input_width, neurons_per_layer, conv_params_per_layer, device)

    # Input layer
    inp = torch.ones((1, 2, 128, 128), dtype=torch.float32).view(1, -1)
    currents, n_currents = module(inp, is_input=True, is_input_and_neurons_combined=False)
    s = inp.view(-1, input_channels, input_height, input_width)
    test_currents = F.conv2d(s, module.conv_filters[0], stride=module.conv_params_per_layer[0].stride, padding=module.conv_params_per_layer[0].padding)
    assert torch.allclose(currents[:, :12288], test_currents.view(1,-1))
    assert currents[:, 12288:].sum() == 0

    # Hidden / Output layer
    inp = torch.ones((1, n_neurons), dtype=torch.float32)
    currents, n_currents = module(inp, is_input=False, is_input_and_neurons_combined=False)
    s_conv = inp[:, 0:12288].view(-1, 3, 64, 64)
    s_fc = inp[:, 12288:14336]
    test_conv_currents = F.conv2d(s_conv, module.conv_filters[1], stride=module.conv_params_per_layer[1].stride, padding=module.conv_params_per_layer[1].padding).view(1, -1)
    test_fc_currents = torch.matmul(s_fc, module.fc_weights.T)
    assert currents[:, :12288].sum() == 0
    assert torch.allclose(currents[:, 12288:14336], test_conv_currents)
    assert torch.allclose(currents[:, 14336:], test_fc_currents)