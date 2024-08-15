from asynctorch.nn.neuron.lif_state import LIFFunction, LIFState
import torch
from snntorch import surrogate
import numpy as np

from asynctorch.utils.surrogate import FastSigmoid

def test_lif_function():
    n_neurons = 10
    batch_size = 2
    torch.manual_seed(0)
    W = torch.rand((n_neurons), dtype=torch.float32, requires_grad=True)
    I_new = torch.ones((batch_size, n_neurons), dtype=torch.float32) * W
    n_I_new = torch.ones((batch_size, n_neurons), dtype=torch.float32)
    membrane_potentials = torch.zeros((batch_size, n_neurons), dtype=torch.float32)
    membrane_threshold = torch.full(size=(n_neurons,), fill_value=0.3, dtype=torch.float32)
    sync_potentials = torch.zeros((batch_size, n_neurons), dtype=torch.float32)
    sync_threshold = torch.zeros(n_neurons, dtype=torch.float32)
    spike_grad = FastSigmoid(1)
    sync_grad = FastSigmoid(1)
    apply_refrac = False
    is_refrac = torch.zeros((batch_size, n_neurons), dtype=torch.bool)
    spk: torch.Tensor = LIFFunction.apply(
        I_new, n_I_new, membrane_potentials, membrane_threshold, sync_potentials, sync_threshold, is_refrac,
        spike_grad, sync_grad, apply_refrac
    )
    spk.sum().backward()
    print(W.grad)

test_lif_function()

def test_zero_current_forward():
    device = torch.device("cpu")
    tau_m = torch.ones(10, device=device)
    membrane_threshold = torch.ones(10, device=device)
    sync_threshold = torch.ones(10, device=device)
    spike_grad = surrogate.atan()

    state_module = LIFState([10], tau_m, membrane_threshold, sync_threshold, spike_grad, spike_grad, device)
    state_module._init_state(2)
    state_module.step_dynamics(100.0)
    I_new = torch.zeros((2, 10), device=device)
    n_I_new = torch.zeros((2, 10), device=device)
    spk = state_module(I_new, n_I_new)
    assert spk.sum() == 0

def test_nonzero_current_forward():
    device = torch.device("cpu")
    tau_m = torch.ones(10, device=device)
    membrane_threshold = torch.full((10,), 1/2, device=device)
    sync_threshold = torch.zeros(10, device=device)
    spike_grad = surrogate.atan()

    state_module = LIFState([10], tau_m, membrane_threshold, sync_threshold, spike_grad, spike_grad, device)
    state_module._init_state(2)
    I_new = torch.ones((2, 10), device=device)
    n_I_new = torch.ones((2, 10), device=device)
    spk = state_module(I_new, n_I_new)
    assert spk.sum() == 20

def test_sync_threshold():
    device = torch.device("cpu")
    tau_m = torch.ones(10, device=device)
    membrane_threshold = torch.full((10,), 1/2, device=device)
    sync_threshold = torch.full((10,), 1.5, device=device)
    spike_grad = surrogate.atan()

    state_module = LIFState([10], tau_m, membrane_threshold, sync_threshold, spike_grad, spike_grad, device)
    state_module._init_state(2)
    I_new = torch.ones((2, 10), device=device)
    n_I_new = torch.ones((2, 10), device=device)
    spk = state_module(I_new, n_I_new)
    assert spk.sum() == 0
    assert state_module.membrane_potentials.sum() == 20
    assert state_module.pre_spike_membrane_potentials.sum() == 0
    I_new = torch.ones((2, 10), device=device)
    n_I_new = torch.ones((2, 10), device=device)
    spk = state_module(I_new, n_I_new)
    assert spk.sum() == 20
    assert state_module.membrane_potentials.sum() == 0
    assert state_module.pre_spike_membrane_potentials.sum() == 40

def test_refrac():
    device = torch.device("cpu")
    tau_m = torch.ones(10, device=device)
    membrane_threshold = torch.full((10,), 1/2, device=device)
    sync_threshold = torch.zeros(10, device=device)
    spike_grad = surrogate.atan()

    state_module = LIFState([10], tau_m, membrane_threshold, sync_threshold, spike_grad, spike_grad, device)
    state_module._init_state(2)
    I_new = torch.ones((2, 10), device=device)
    n_I_new = torch.ones((2, 10), device=device)
    spk = state_module(I_new, n_I_new)
    assert spk.sum() == 20
    I_new = torch.ones((2, 10), device=device)
    n_I_new = torch.ones((2, 10), device=device)
    spk = state_module(I_new, n_I_new)
    assert spk.sum() == 0
    state_module.step_dynamics(1.0)
    I_new = torch.ones((2, 10), device=device)
    n_I_new = torch.ones((2, 10), device=device)
    spk = state_module(I_new, n_I_new)
    assert spk.sum() == 20

def test_membrane_decay():
    device = torch.device("cpu")
    tau_m = torch.ones(10, device=device)
    membrane_threshold = torch.full((10,), 1.00000001, device=device)
    sync_threshold = torch.zeros((10,), device=device)
    spike_grad = surrogate.atan()

    state_module = LIFState([10], tau_m, membrane_threshold, sync_threshold, spike_grad, spike_grad, device)
    state_module._init_state(2)
    I_new = torch.ones((2, 10), device=device)
    n_I_new = torch.ones((2, 10), device=device)
    spk = state_module(I_new, n_I_new)
    assert spk.sum() == 0
    state_module.step_dynamics(1.0)
    assert np.allclose(state_module.membrane_potentials[0,:], 1*np.exp(-1.0/tau_m))
    assert np.allclose(state_module.membrane_potentials[1,:], 1*np.exp(-1.0/tau_m))
    state_module.step_dynamics(1.0)
    assert np.allclose(state_module.membrane_potentials[0,:], 1*np.exp(-2.0/tau_m))
    assert np.allclose(state_module.membrane_potentials[1,:], 1*np.exp(-2.0/tau_m))
    I_new = torch.ones((2, 10), device=device)
    n_I_new = torch.ones((2, 10), device=device)
    spk = state_module(I_new, n_I_new)
    assert spk.sum() == 20
    assert state_module.membrane_potentials.sum() == 0