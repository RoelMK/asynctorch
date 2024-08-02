from asynctorch.nn.architecture.sparse_fully_linear_architecture import SparseFullyLinearArchitecture
import torch
import torch.nn as nn


def test_init_module():
    n_inputs = 2
    neurons_per_layer = [3, 4]
    device = torch.device("cpu")
    module = SparseFullyLinearArchitecture(n_inputs, neurons_per_layer, device)
    state_dict = module.state_dict()
    assert state_dict["weights"].shape == (7, 9)
    assert state_dict["weights_mask"].shape == (7, 9)
    assert state_dict["weights_mask"].sum() == (2*3+3*4)
    assert state_dict["weights_mask"][:, 0:2].sum() == 6
    assert state_dict["weights_mask"][:, 2:5].sum() == 12

def test_load_and_forward():
    n_inputs = 2
    neurons_per_layer = [3, 4]
    device = torch.device("cpu")
    module = SparseFullyLinearArchitecture(n_inputs, neurons_per_layer, device)
    weights_per_layer = [
        torch.ones((3, 2), dtype=torch.float32),
        torch.ones((4, 3), dtype=torch.float32),
    ]
    state_dict = SparseFullyLinearArchitecture.create_state_dict_from_weights_list(weights_per_layer)
    module.load_state_dict(state_dict)
    assert state_dict["weights"].shape == (7, 9)
    assert state_dict["weights_mask"][:, 0:2].sum() == 6
    assert state_dict["weights_mask"][:, 2:5].sum() == 12
    assert state_dict["weights"][:, 0:2].sum() == 6
    assert state_dict["weights"][:, 2:5].sum() == 12
    s_in = torch.ones((2, 7), dtype=torch.float32)
    currents, n_currents = module(s_in, is_input=False, is_input_and_neurons_combined=False)
    assert currents.shape == (2, 7)
    assert n_currents.shape == (2, 7)
    assert currents.sum() == 24
    assert n_currents.sum() == 24

def test_forward_combined_input_and_neurons():
    n_inputs = 2
    neurons_per_layer = [3, 4]
    device = torch.device("cpu")
    module = SparseFullyLinearArchitecture(n_inputs, neurons_per_layer, device)
    weights_per_layer = [
        torch.ones((3, 2), dtype=torch.float32),
        torch.ones((4, 3), dtype=torch.float32),
    ]
    state_dict = SparseFullyLinearArchitecture.create_state_dict_from_weights_list(weights_per_layer)
    module.load_state_dict(state_dict)
    s_in = torch.ones((2, 9), dtype=torch.float32)
    currents, n_currents = module(s_in, is_input=False, is_input_and_neurons_combined=True)
    assert currents.shape == (2, 7)
    assert n_currents.shape == (2, 7)
    assert currents.sum() == 36
    assert n_currents.sum() == 36

def test_forward_input():
    n_inputs = 2
    neurons_per_layer = [3, 4]
    device = torch.device("cpu")
    module = SparseFullyLinearArchitecture(n_inputs, neurons_per_layer, device)
    weights_per_layer = [
        torch.ones((3, 2), dtype=torch.float32),
        torch.ones((4, 3), dtype=torch.float32),
    ]
    state_dict = SparseFullyLinearArchitecture.create_state_dict_from_weights_list(weights_per_layer)
    module.load_state_dict(state_dict)
    s_in = torch.ones((2, 2), dtype=torch.float32)
    currents, n_currents = module(s_in, is_input=True, is_input_and_neurons_combined=False)
    assert currents.shape == (2, 7)
    assert n_currents.shape == (2, 7)
    assert currents.sum() == 12
    assert n_currents.sum() == 12

def test_linear_equivalence():
    n_inputs = 2
    neurons_per_layer = [3]
    device = torch.device("cpu")
    module = SparseFullyLinearArchitecture(n_inputs, neurons_per_layer, device)
    weights_per_layer = [
        torch.rand(3, 2, dtype=torch.float32),
    ]
    state_dict = SparseFullyLinearArchitecture.create_state_dict_from_weights_list(weights_per_layer)
    module.load_state_dict(state_dict)
    s_in = torch.ones((1, 2), dtype=torch.float32)
    currents, _ = module(s_in, is_input=True, is_input_and_neurons_combined=False)
    linear = nn.Linear(2, 3, bias=False)
    linear.weight.data = weights_per_layer[0]
    currents_linear = linear(s_in)
    assert torch.allclose(currents, currents_linear)
