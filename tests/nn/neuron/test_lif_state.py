from asynctorch.nn.neuron.lif_state import LIFFunction, LIFState
import torch
import numpy as np
import time

from asynctorch.utils.surrogate import FastSigmoid
from torch.profiler import profile, record_function, ProfilerActivity

def test_lif_function():
    n_neurons = 1000
    batch_size = 2
    torch.manual_seed(0)
    W = torch.rand((n_neurons), dtype=torch.float32, requires_grad=True)
    s_in = torch.ones((batch_size, n_neurons), dtype=torch.float32)
    s_in[:, :n_neurons-10] = 0
    I_new = s_in * W
    n_I_new = s_in
    membrane_potentials = torch.zeros((batch_size, n_neurons), dtype=torch.float32)
    membrane_threshold = torch.full(size=(n_neurons,), fill_value=0.3, dtype=torch.float32)
    sync_potentials = torch.zeros((batch_size, n_neurons), dtype=torch.float32)
    sync_threshold = torch.zeros(n_neurons, dtype=torch.float32)
    spike_grad = FastSigmoid(1)
    sync_grad = FastSigmoid(1)
    update_mask = I_new != 0

    membrane_potentials = membrane_potentials + I_new
    sync_potentials = sync_potentials + n_I_new

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        t = time.time()
        spk = LIFFunction.apply(
            update_mask, membrane_potentials, membrane_threshold, sync_potentials, sync_threshold, spike_grad, sync_grad
        )
        print('Forward time: ', time.time() - t)
        t = time.time()
        spk.sum().backward()
        print('Backward time: ', time.time() - t)
    #print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

def test_zero_current_forward():
    device = torch.device("cpu")
    tau_m = torch.ones(10, device=device)
    membrane_threshold = torch.ones(10, device=device)
    sync_threshold = torch.ones(10, device=device)
    spike_grad = FastSigmoid(1)

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
    spike_grad = FastSigmoid(1)

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
    spike_grad = FastSigmoid(1)

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
    spike_grad = FastSigmoid(1)

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
    spike_grad = FastSigmoid(1)

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