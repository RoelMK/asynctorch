from typing import List, Union
import torch
import torch.nn as nn
from asynctorch.nn.neuron.neuron_state import NeuronState
from asynctorch.utils.surrogate import SurrogateThresholdFunction
import functools

class LIFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                membrane_potentials: torch.Tensor, 
                membrane_threshold: torch.Tensor, 
                sync_potentials: torch.Tensor, 
                sync_threshold: torch.Tensor, 
                spike_grad: SurrogateThresholdFunction, 
                sync_grad: SurrogateThresholdFunction):
        check_membrane = (membrane_potentials - membrane_threshold)
        check_sync = (sync_potentials - sync_threshold)
        ctx.save_for_backward(check_membrane, check_sync)
        ctx.spike_grad = spike_grad
        ctx.sync_grad = sync_grad

        # >> Check thresholds and spike
        spk: torch.Tensor = spike_grad.forward(check_membrane) * sync_grad.forward(check_sync)
        return spk
    
    @staticmethod
    def backward(ctx, dL_dspk: torch.Tensor):
        check_membrane, check_sync = ctx.saved_tensors
        spike_grad = ctx.spike_grad
        sync_grad = ctx.sync_grad

        dL_dmembrane_potentials = dL_dspk * spike_grad.backward(check_membrane)
        dL_dsync_potentials = dL_dspk * sync_grad.backward(check_sync)

        return dL_dmembrane_potentials, None, dL_dsync_potentials, None, None, None, None

    
    
class LIFState(NeuronState):
    membrane_potentials: torch.Tensor  # shape = (batch_size, n_neurons)
    pre_spike_membrane_potentials: torch.Tensor  # shape = (batch_size, n_neurons)
    is_refrac: torch.Tensor  # shape = (batch_size, n_neurons)
    synchronization_potentials: torch.Tensor  # shape = (batch_size, n_neurons)

    def __init__(
        self,
        neurons_per_layer: List[int],
        tau_m: Union[torch.Tensor, float],
        membrane_threshold: Union[torch.Tensor, float],
        sync_threshold: Union[torch.Tensor, float],
        spike_grad,
        sync_grad,
        device: torch.device,
        *args,
        apply_refrac: bool = True,
        refrac_dropout: float = 0.0,
        trainable_sync_threshold: bool = False,
        save_grad_sparse = False, # Saves gradients as sparse tensors
        **kwargs,
    ):
        super().__init__(neurons_per_layer, device, *args, **kwargs)
        if not isinstance(tau_m, torch.Tensor):
            tau_m = torch.full((self.n_neurons,), tau_m, dtype=torch.float, device=device)
        if not isinstance(membrane_threshold, torch.Tensor):
            membrane_threshold = torch.full((self.n_neurons,), membrane_threshold, dtype=torch.float, device=device)
        if not isinstance(sync_threshold, torch.Tensor):
            sync_threshold = torch.full((self.n_neurons,), sync_threshold, dtype=torch.float, device=device)

        if not torch.all(tau_m > 0):
            raise ValueError("tau_m must be positive")
        if len(membrane_threshold.shape) != 1:
            raise ValueError("membrane_threshold must be 1D")
        if tau_m.shape != membrane_threshold.shape or tau_m.shape != sync_threshold.shape:
            raise ValueError("tau_m, membrane_threshold and sync_threshold must have same shape")
        
        self.membrane_threshold = membrane_threshold
        if trainable_sync_threshold:
            self.sync_threshold = nn.Parameter(sync_threshold)
            grad_mask_sync_threshold = torch.zeros((self.n_neurons,), dtype=torch.bool, device=self.device)
            grad_mask_sync_threshold[self.n_inputs :] = True
            self.sync_threshold.register_hook(lambda grad: grad * grad_mask_sync_threshold)
        else:
            self.sync_threshold = sync_threshold
        self.tau_m_inversed = tau_m.pow(-1)
        self.spike_grad = spike_grad  # surrogate gradient for spike function
        self.sync_grad = sync_grad  # surrogate gradient for synchronization function
        self.refrac_dropout = refrac_dropout
        self.apply_refrac = apply_refrac
        self.save_grad_sparse = save_grad_sparse
        self._reset_state()

    def is_init(self) -> bool:
        return self.membrane_potentials is not None

    def _init_state(self, batch_size: int):
        if self.is_init():
            raise RuntimeError("Cannot initialize state twice")
        self.membrane_potentials = torch.zeros((batch_size, self.n_neurons), dtype=torch.float, device=self.device)
        if self.apply_refrac:
            self.is_refrac = torch.zeros_like(self.membrane_potentials, dtype=torch.bool)
        self.synchronization_potentials = torch.zeros_like(self.membrane_potentials, dtype=torch.float)
        self.pre_spike_membrane_potentials = torch.zeros_like(self.membrane_potentials, dtype=torch.float)

    def _reset_state(self):
        self.membrane_potentials = None
        if self.apply_refrac:
            self.is_refrac = None
        self.synchronization_potentials = None
        self.pre_spike_membrane_potentials = None

    def _detach_state(self):
        if self.is_init():
            self.membrane_potentials = self.membrane_potentials.detach()
            self.synchronization_potentials = self.synchronization_potentials.detach()
            if self.apply_refrac:
                self.is_refrac = self.is_refrac.detach()
            self.pre_spike_membrane_potentials = self.pre_spike_membrane_potentials.detach()

    def get_pre_spike_state(self) -> torch.Tensor:
        return self.pre_spike_membrane_potentials
    
    def get_post_spike_state(self) -> torch.Tensor:
        return self.membrane_potentials

    def step_dynamics(self, dt: float):
        # >> Decay membrane potential
        beta = torch.exp(-dt * self.tau_m_inversed)
        self.membrane_potentials = self.membrane_potentials * beta
        # >> Reset refractory period, synchronization potential, and pre-spike membrane potential
        self.synchronization_potentials = torch.zeros_like(self.membrane_potentials, dtype=torch.float)
        if self.apply_refrac:
            self.is_refrac = torch.zeros_like(self.membrane_potentials, dtype=torch.bool)
        self.pre_spike_membrane_potentials = torch.zeros_like(self.membrane_potentials, dtype=torch.float)

    def forward(self, I_new: torch.Tensor, n_I_new: torch.Tensor) -> torch.Tensor:
        if not self.is_init():
            raise RuntimeError("State module not initialized")

        def sparse_pack_hook(mask, x):
            return (x * mask).to_sparse()
        
        def sparse_unpack_hook(x):
            return x.to_dense()
        
        def identity_hook(x):
            return x

        if self.save_grad_sparse:
            with torch.no_grad():
                update_mask = I_new != 0
            pack_hook = functools.partial(sparse_pack_hook, update_mask)
            unpack_hook = sparse_unpack_hook
        else:
            pack_hook = identity_hook
            unpack_hook = identity_hook
        
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            # >> Add current, apply refractory period
            if self.apply_refrac:
                membrane_potentials = (self.membrane_potentials + I_new) * ~self.is_refrac
            else:
                membrane_potentials = self.membrane_potentials + I_new
            self.synchronization_potentials = self.synchronization_potentials + n_I_new

            # >> Check thresholds and spike
            spk = LIFFunction.apply(
                membrane_potentials, self.membrane_threshold, self.synchronization_potentials, self.sync_threshold, self.spike_grad, self.sync_grad
            )

        with torch.no_grad():
            # >> Update membrane potential
            self.membrane_potentials = membrane_potentials * (-spk + 1)
            self.pre_spike_membrane_potentials = membrane_potentials * spk + self.pre_spike_membrane_potentials * (1 - spk)
            # >> Set refractory period
            if self.apply_refrac:
                if self.refrac_dropout > 0:
                    refrac_add = torch.where(torch.rand_like(spk) < self.refrac_dropout, torch.zeros_like(spk), spk)
                else:
                    refrac_add = spk
                self.is_refrac.add_(refrac_add.bool())

        return spk
