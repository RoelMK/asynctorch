from asynctorch.nn.architecture.semi_sparse_fully_linear_architecture import SemiSparseFullyLinearArchitecture
import torch
import torch.nn as nn


def test_init_module():
    n_inputs = 2
    neurons_per_layer = [3, 4]
    device = torch.device("cpu")
    module = SemiSparseFullyLinearArchitecture(n_inputs, neurons_per_layer, device)
    state_dict = module.state_dict()
    assert state_dict["weights"].shape == (7, 7)
    assert state_dict["weights_mask"].shape == (7, 7)
    assert state_dict["weights_mask"].sum() == 3*4
    assert state_dict["weights_mask"][:, 0:3].sum() == 12

def test_load_and_forward():
    n_inputs = 2
    neurons_per_layer = [3, 4]
    device = torch.device("cpu")
    module = SemiSparseFullyLinearArchitecture(n_inputs, neurons_per_layer, device)
    weights_per_layer = [
        torch.ones((3, 2), dtype=torch.float32),
        torch.ones((4, 3), dtype=torch.float32),
    ]
    state_dict = SemiSparseFullyLinearArchitecture.create_state_dict_from_weights_list(weights_per_layer)
    module.load_state_dict(state_dict)
    assert state_dict["weights"].shape == (7, 7)
    assert state_dict["weights_mask"][:, 0:3].sum() == 3*4
    assert state_dict["weights"][:, 0:3].sum() == 12
    s_in = torch.ones((2, 7), dtype=torch.float32)
    currents, n_currents = module(s_in, is_input=False, is_input_and_neurons_combined=False)
    assert currents.shape == (2, 7)
    assert n_currents.shape == (2, 7)
    assert currents.sum() == 24
    assert n_currents.sum() == 24

def test_forward_input():
    n_inputs = 2
    neurons_per_layer = [3, 4]
    device = torch.device("cpu")
    module = SemiSparseFullyLinearArchitecture(n_inputs, neurons_per_layer, device)
    weights_per_layer = [
        torch.ones((3, 2), dtype=torch.float32),
        torch.ones((4, 3), dtype=torch.float32),
    ]
    state_dict = SemiSparseFullyLinearArchitecture.create_state_dict_from_weights_list(weights_per_layer)
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
    module = SemiSparseFullyLinearArchitecture(n_inputs, neurons_per_layer, device)
    weights_per_layer = [
        torch.rand(3, 2, dtype=torch.float32),
    ]
    state_dict = SemiSparseFullyLinearArchitecture.create_state_dict_from_weights_list(weights_per_layer)
    module.load_state_dict(state_dict)
    s_in = torch.ones((1, 2), dtype=torch.float32)
    currents, _ = module(s_in, is_input=True, is_input_and_neurons_combined=False)
    linear = nn.Linear(2, 3, bias=False)
    linear.weight.data = weights_per_layer[0]
    currents_linear = linear(s_in)
    assert torch.allclose(currents, currents_linear)
