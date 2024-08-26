import torch
import torch.nn as nn
from asynctorch.nn.neuron.lif_state import LIFState
from asynctorch.utils.surrogate import SurrogateThresholdFunction


#### This is work in progress, not yet tested ####

class SynchronizingLIFFunction(torch.autograd.Function):
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

        dspk_dmem = spike_grad.backward(check_membrane)
        dspk_dsync = sync_grad.backward(check_sync)

        dL_dmembrane_potentials = dL_dspk * dspk_dmem
        dL_dsync_potentials = dL_dspk * dspk_dsync

        return dL_dmembrane_potentials, None, dL_dsync_potentials, None, None, None, None

    
    
class SynchronizingLIFState(LIFState):
    synchronization_potentials: torch.Tensor  # shape = (batch_size, n_neurons)

    def __init__(
        self,
        sync_grad,
        device: torch.device,
        *args,
        trainable_sync_threshold: bool = False,
        **kwargs,
    ):
        super().__init__(device, *args, **kwargs)
        if not isinstance(sync_threshold, torch.Tensor):
            sync_threshold = torch.full((self.n_neurons,), sync_threshold, dtype=torch.float, device=device)

        if self.membrane_threshold.shape != sync_threshold.shape:
            raise ValueError(" membrane_threshold and sync_threshold must have same shape")
        
        if self.backprop_threshold is not None:
            raise ValueError(" backprop_threshold is not supported for SynchronizingLIFState")
        
        if trainable_sync_threshold:
            self.sync_threshold = nn.Parameter(sync_threshold)
            grad_mask_sync_threshold = torch.zeros((self.n_neurons,), dtype=torch.bool, device=self.device)
            grad_mask_sync_threshold[self.n_inputs :] = True
            self.sync_threshold.register_hook(lambda grad: grad * grad_mask_sync_threshold)
        else:
            self.sync_threshold = sync_threshold
        self.sync_grad = sync_grad  # surrogate gradient for synchronization function
        self._reset_state()

    def _init_state(self, batch_size: int):
        super()._init_state(batch_size)
        self.synchronization_potentials = torch.zeros_like(self.membrane_potentials, dtype=torch.float)

    def _reset_state(self):
        super()._reset_state()
        self.synchronization_potentials = None

    def _detach_state(self):
        super()._detach_state()
        if self.is_init():
            self.synchronization_potentials = self.synchronization_potentials.detach()

    def step_dynamics(self, dt: float):
        super().step_dynamics(dt)
        self.synchronization_potentials = torch.zeros_like(self.membrane_potentials, dtype=torch.float)

    def forward(self, I_new: torch.Tensor, n_I_new: torch.Tensor) -> torch.Tensor:
        # TODO: share more code with LIFState.forward
        if not self.is_init():
            raise RuntimeError("State module not initialized")

        # >> Add current, apply refractory period
        if self.apply_refrac:
            membrane_potentials = (self.membrane_potentials + I_new) * ~self.is_refrac
        else:
            membrane_potentials = self.membrane_potentials + I_new
        self.synchronization_potentials = self.synchronization_potentials + n_I_new

        # >> Check thresholds and spike
        spk = SynchronizingLIFFunction.apply(
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
