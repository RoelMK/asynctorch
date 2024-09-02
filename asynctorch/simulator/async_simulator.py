from typing import List
from asynctorch.simulator.extensions.forward_step_extension import ForwardStepExtension
from asynctorch.simulator.spike_selector import SpikeSelector
from asynctorch.nn.neuron.neuron_state import NeuronState
import torch
import torch.nn as nn


class AsyncSimulator(nn.Module):
    def __init__(
        self,
        neuron_state: NeuronState,
        spike_selector: SpikeSelector,
        forward_step_extensions: List[ForwardStepExtension] = [],
    ):
        super().__init__()
        self.neuron_state = neuron_state
        self.spike_selector = spike_selector
        self.device = neuron_state.device
        self.forward_step_extensions = forward_step_extensions
        self.n_neurons = self.neuron_state.n_neurons
        self.reset_state()

    def is_init(self) -> bool:
        return self.spike_counts is not None and self.neuron_state.is_init() and self.spike_selector.is_init()

    def init_state(self, batch_size: int):
        if self.is_init():
            raise RuntimeError("Cannot initialize state twice")
        self.neuron_state._init_state(batch_size)
        self.spike_selector._init_state(batch_size)
        self.spike_counts = torch.zeros((batch_size, self.n_neurons), dtype=torch.int64, device=self.device)
        for extension in self.forward_step_extensions:
            extension._init_state(batch_size)

    def reset_state(self):
        self.n_forward_steps = 0
        self.neuron_state._reset_state()
        self.spike_selector._reset_state()
        self.spike_counts = None
        for extension in self.forward_step_extensions:
            extension._reset_state()

    def detach_state(self):
        self.neuron_state._detach_state()
        self.spike_selector._detach_state()
        if self.spike_counts is not None:
            self.spike_counts = self.spike_counts.detach()
        for extension in self.forward_step_extensions:
            extension._detach_state()

    def forward(self, x: torch.Tensor, dt: float, is_current: bool = False) -> torch.Tensor:
        # Check input
        batch_size = x.shape[0]
        if not self.is_init():
            self.init_state(batch_size)
        elif self.spike_counts.shape[0] != batch_size:
            raise ValueError(
                f"batch_size of input ({batch_size}) must match batch_size of previous input ({self.spike_counts.shape[0]})"
            )
        spike_counts = torch.zeros((batch_size, self.n_neurons), dtype=torch.float32, device=self.device)

        # Pass input
        if not is_current:
            for extension in self.forward_step_extensions:
                extension.on_spike(x, spike_counts, self.spike_counts, is_input=True)
        self.neuron_state.step_dynamics(dt)
        I_new, n_I_new = self.spike_selector(x, is_input=True)
        for extension in self.forward_step_extensions:
            extension.on_current(I_new, n_I_new, is_input=True)
        s_new: torch.Tensor = None

        # Process spikes
        while (not self.spike_selector.is_done() or s_new is None or torch.any(I_new > 0)) and not any(
            extension.force_stop() for extension in self.forward_step_extensions
        ):
            s_new = self.neuron_state(I_new, n_I_new)
            for extension in self.forward_step_extensions:
                s_new = extension.on_spike(s_new, spike_counts, self.spike_counts, is_input=False)
            spike_counts = spike_counts + s_new
            I_new, n_I_new = self.spike_selector(s_new)
            for extension in self.forward_step_extensions:
                I_new, n_I_new = extension.on_current(I_new, n_I_new, is_input=False)
            self.n_forward_steps += 1
        self.spike_counts = self.spike_counts + spike_counts

        for extension in self.forward_step_extensions:
            extension.on_done(self.spike_counts)

        return spike_counts
