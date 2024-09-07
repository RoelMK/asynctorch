from typing import List, Union
import torch
from asynctorch.nn.neuron.neuron_state import NeuronState
from asynctorch.utils.surrogate import SurrogateThresholdFunction

class LIFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                membrane_potentials: torch.Tensor, 
                membrane_threshold: torch.Tensor, 
                spike_grad: SurrogateThresholdFunction, 
                backprop_threshold: Union[float, None]): # None means no thresholding
        check_membrane = (membrane_potentials - membrane_threshold)

        if backprop_threshold is not None:
            mask_membrane = check_membrane > backprop_threshold
            check_membrane_backwards = check_membrane * mask_membrane
            ctx.save_for_backward(check_membrane_backwards.to_sparse())
        else:
            ctx.save_for_backward(check_membrane)
        ctx.spike_grad = spike_grad

        # >> Check thresholds and spike
        spk: torch.Tensor = spike_grad.forward(check_membrane)
        return spk
    
    @staticmethod
    def backward(ctx, dL_dspk: torch.Tensor):
        check_membrane, = ctx.saved_tensors
        spike_grad = ctx.spike_grad

        if check_membrane.is_sparse:
            dspk_dmem = torch.zeros_like(dL_dspk)
            dspk_dmem[check_membrane._indices().unbind()] = spike_grad.backward(check_membrane._values())
        else:
            dspk_dmem = spike_grad.backward(check_membrane)

        dL_dmembrane_potentials = dL_dspk * dspk_dmem

        return dL_dmembrane_potentials, None, None, None

    
    
class LIFState(NeuronState):
    membrane_potentials: torch.Tensor  # shape = (batch_size, n_neurons)
    pre_spike_membrane_potentials: torch.Tensor  # shape = (batch_size, n_neurons)
    is_refrac: torch.Tensor  # shape = (batch_size, n_neurons)

    def __init__(
        self,
        neurons_per_layer: List[int],
        tau_m: Union[torch.Tensor, float],
        membrane_threshold: Union[torch.Tensor, float],
        spike_grad,
        device: torch.device,
        *args,
        apply_refrac: bool = True,
        refrac_dropout: float = 0.0,
        backprop_threshold = None, 
        **kwargs,
    ):
        super().__init__(neurons_per_layer, device, *args, **kwargs)
        if not isinstance(tau_m, torch.Tensor):
            tau_m = torch.full((self.n_neurons,), tau_m, dtype=torch.float, device=device)
        if not isinstance(membrane_threshold, torch.Tensor):
            membrane_threshold = torch.full((self.n_neurons,), membrane_threshold, dtype=torch.float, device=device)

        if not torch.all(tau_m > 0):
            raise ValueError("tau_m must be positive")
        if len(membrane_threshold.shape) != 1:
            raise ValueError("membrane_threshold must be 1D")
        if tau_m.shape != membrane_threshold.shape:
            raise ValueError("tau_m and membrane_threshold must have same shape")
        
        self.membrane_threshold = membrane_threshold
        self.tau_m_inversed = tau_m.pow(-1)
        self.spike_grad = spike_grad  # surrogate gradient for spike function
        self.refrac_dropout = refrac_dropout
        self.apply_refrac = apply_refrac
        self.backprop_threshold = backprop_threshold
        self._reset_state()

    def is_init(self) -> bool:
        return self.membrane_potentials is not None

    def _init_state(self, batch_size: int):
        if self.is_init():
            raise RuntimeError("Cannot initialize state twice")
        self.membrane_potentials = torch.zeros((batch_size, self.n_neurons), dtype=torch.float, device=self.device)
        if self.apply_refrac:
            self.is_refrac = torch.zeros_like(self.membrane_potentials, dtype=torch.bool)
        self.pre_spike_membrane_potentials = torch.zeros_like(self.membrane_potentials, dtype=torch.float)

    def _reset_state(self):
        self.membrane_potentials = None
        if self.apply_refrac:
            self.is_refrac = None
        self.pre_spike_membrane_potentials = None

    def _detach_state(self):
        if self.is_init():
            self.membrane_potentials = self.membrane_potentials.detach()
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
        # >> Reset refractory period, and pre-spike membrane potential
        if self.apply_refrac:
            self.is_refrac = torch.zeros_like(self.membrane_potentials, dtype=torch.bool)
        self.pre_spike_membrane_potentials = torch.zeros_like(self.membrane_potentials, dtype=torch.float)

    def forward(self, I_new: torch.Tensor, n_I_new: torch.Tensor) -> torch.Tensor:
        if not self.is_init():
            raise RuntimeError("State module not initialized")

        # >> Add current, apply refractory period
        if self.apply_refrac:
            membrane_potentials = (self.membrane_potentials + I_new) * ~self.is_refrac
        else:
            membrane_potentials = self.membrane_potentials + I_new

        # >> Check thresholds and spike
        spk = LIFFunction.apply(
            membrane_potentials, self.membrane_threshold, self.spike_grad, self.backprop_threshold
        )

        # >> Update membrane potential
        self.membrane_potentials = membrane_potentials * (-spk + 1)
        with torch.no_grad():
            self.pre_spike_membrane_potentials = membrane_potentials * spk + self.pre_spike_membrane_potentials * (1 - spk)
            # >> Set refractory period
            if self.apply_refrac:
                if self.refrac_dropout > 0:
                    refrac_add = torch.where(torch.rand_like(spk) < self.refrac_dropout, torch.zeros_like(spk), spk)
                else:
                    refrac_add = spk
                self.is_refrac.add_(refrac_add.bool())

        return spk
