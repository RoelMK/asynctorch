import torch
from torch import nn
import numpy as np
from asynctorch.nn.architecture.mixed_architecture import AsyncLayer, AsyncNetwork

def test_fc_async_layer():
    linear = nn.Linear(2, 2)
    n_neurons = 4
    layer = AsyncLayer(linear, [0, 1], [2, 3], n_neurons)
    inputs = torch.ones(size=(1, n_neurons))
    outputs, n_outputs = layer(inputs)
    assert outputs.shape == (1, n_neurons)
    linear_outputs = linear(inputs[:, [0, 1]])
    assert outputs[0, 2] == linear_outputs[0, 0]
    assert outputs[0, 3] == linear_outputs[0, 1]

def test_conv_async_layer():
    conv = nn.Conv2d(2, 2, kernel_size=3, padding=1)
    n_neurons = 2*3*3 + 2*3*3
    layer = AsyncLayer(conv, list(range(0, 2*3*3)), list(range(2*3*3, n_neurons)), n_neurons, reshape_input_to=(2, 3, 3))
    inputs = torch.ones(size=(1, n_neurons))
    outputs, n_outputs = layer(inputs)
    assert outputs.shape == (1, n_neurons)
    conv_outputs = conv(inputs[:, :2*3*3].view(1, 2, 3, 3))
    assert torch.allclose(outputs[0, 2*3*3:], conv_outputs.view(1, -1))

def test_async_network():
    input_shape = (1, 9, 9)
    layer1_shape = (2, 9, 9)
    layer2_shape = (64, 9, 9)
    output_shape = (10,)
    n_neurons = np.prod(layer1_shape) + np.prod(layer2_shape) + np.prod(output_shape)
    input_layer = AsyncLayer(nn.Conv2d(1, 2, kernel_size=3, padding=1), 
                             list(range(0, np.prod(input_shape))), 
                             list(range(np.prod(layer1_shape))), 
                             n_neurons, 
                             reshape_input_to=input_shape)
    layer1 = AsyncLayer(nn.Conv2d(2, 64, kernel_size=3, padding=1),
                        list(range(0, np.prod(layer1_shape))),
                        list(range(np.prod(layer1_shape), np.prod(layer1_shape) + np.prod(layer2_shape))),
                        n_neurons,
                        reshape_input_to=layer1_shape)
    layer2 = AsyncLayer(nn.Linear(np.prod(layer2_shape), 10),
                        list(range(np.prod(layer1_shape), np.prod(layer1_shape) + np.prod(layer2_shape))),
                        list(range(np.prod(layer1_shape) + np.prod(layer2_shape), n_neurons)),
                        n_neurons)
    async_network = AsyncNetwork(input_layer, [layer1, layer2], n_neurons)
    batch_size = 10
    # External input
    inputs = torch.ones(size=(batch_size, 1, 9, 9), dtype=torch.float32)
    currents, n_currents = async_network(inputs.view(batch_size, -1), is_input=True)
    assert currents.shape == (batch_size, n_neurons)
    seq_currents = input_layer.module(inputs).view(batch_size, -1)
    assert torch.allclose(currents[:, :np.prod(layer1_shape)], seq_currents)
    # Network input
    inputs = torch.ones(size=(batch_size, n_neurons), dtype=torch.float32)
    currents, n_currents = async_network(inputs, is_input=False)
    assert currents.shape == (batch_size, n_neurons)
    seq_currents_layer1 = layer1.module(inputs[:, :np.prod(layer1_shape)].view(batch_size, *layer1_shape)).view(batch_size, -1)
    assert torch.allclose(currents[:, np.prod(layer1_shape):np.prod(layer1_shape) + np.prod(layer2_shape)], seq_currents_layer1)
    seq_currents_layer2 = layer2.module(inputs[:, np.prod(layer1_shape):np.prod(layer1_shape) + np.prod(layer2_shape)]).view(batch_size, -1)
    assert torch.allclose(currents[:, np.prod(layer1_shape) + np.prod(layer2_shape):], seq_currents_layer2)
