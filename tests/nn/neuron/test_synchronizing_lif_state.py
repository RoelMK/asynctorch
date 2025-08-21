# TODO: work on this test
# def test_sync_threshold():
#     device = torch.device("cpu")
#     tau_m = torch.ones(10, device=device)
#     membrane_threshold = torch.full((10,), 1/2, device=device)
#     spike_grad = FastSigmoid(1)

#     state_module = LIFState([10], tau_m, membrane_threshold, spike_grad, device)
#     state_module._init_state(2)
#     I_new = torch.ones((2, 10), device=device)
#     n_I_new = torch.ones((2, 10), device=device)
#     spk = state_module(I_new, n_I_new)
#     assert spk.sum() == 0
#     assert state_module.membrane_potentials.sum() == 20
#     assert state_module.pre_spike_membrane_potentials.sum() == 0
#     I_new = torch.ones((2, 10), device=device)
#     n_I_new = torch.ones((2, 10), device=device)
#     spk = state_module(I_new, n_I_new)
#     assert spk.sum() == 20
#     assert state_module.membrane_potentials.sum() == 0
#     assert state_module.pre_spike_membrane_potentials.sum() == 40