from abc import ABC, abstractmethod
from typing import Callable
import torch

from asynctorch.nn.neuron.neuron_state import NeuronState

class SpikeScheduler(ABC):
    def __init__(self, neuron_state: NeuronState):
        self.neuron_state = neuron_state

    @abstractmethod
    def sort(self, outgoing_spikes: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        pass


class RandomSpikeScheduler(SpikeScheduler):
    def __init__(self, neuron_state: NeuronState):
        super().__init__(neuron_state)

    def sort(self, outgoing_spikes: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        nonzero_mask = outgoing_spikes != 0
        random_values = torch.rand_like(outgoing_spikes, dtype=torch.float)
        max_random_value = random_values.max() + 1
        random_values_masked = torch.where(nonzero_mask, random_values, max_random_value)
        sorted_indices = random_values_masked.argsort(dim=1)
        return indices.gather(1, sorted_indices)


class MomentumSpikeScheduler(SpikeScheduler):
    def __init__(self, neuron_state: NeuronState, lambda_: float = 1e-6):
        super().__init__(neuron_state)
        self.lambda_ = lambda_
        
    def sort(self, outgoing_spikes: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        nonzero_mask = outgoing_spikes != 0
        pre_spike_state = self.neuron_state.get_pre_spike_state()
        min_random_value = pre_spike_state.min() - 1
        state_masked = torch.where(nonzero_mask, pre_spike_state, min_random_value)
        state_masked += torch.rand_like(state_masked) * self.lambda_ * nonzero_mask
        sorted_indices = state_masked.argsort(dim=1, descending=True)
        return indices.gather(1, sorted_indices)
    

class NoiseScheduler(ABC):
    def __init__(self):
        self.n = 0

    @abstractmethod
    def get_lambda(self) -> float:
        pass

    def step(self):
        self.n += 1

    def reset(self):
        self.n = 0

class CallableNoiseScheduler(NoiseScheduler):
    def __init__(self, callable: Callable[[int], float]):
        super().__init__()
        self.callable = callable

    def get_lambda(self) -> float:
        return self.callable(self.n)
    
class StepNoiseScheduler(NoiseScheduler):
    def __init__(self, start_lambda: float, step_size: int, gamma: float, last_epoch: int = 0):
        super().__init__()
        self.start_lambda = start_lambda
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch

    def get_lambda(self) -> float:
        return self.start_lambda * (self.gamma ** ((self.n - self.last_epoch) // self.step_size))

class ScheduledMomentumSpikeScheduler(SpikeScheduler):
    def __init__(self, neuron_state: NeuronState, noise_scheduler: NoiseScheduler):
        super().__init__(neuron_state)
        self.noise_scheduler = noise_scheduler
        
    def sort(self, outgoing_spikes: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        nonzero_mask = outgoing_spikes != 0
        pre_spike_state = self.neuron_state.get_pre_spike_state()
        min_random_value = pre_spike_state.min() - 1
        state_masked = torch.where(nonzero_mask, pre_spike_state, min_random_value)
        state_masked += torch.rand_like(state_masked) * self.noise_scheduler.get_lambda() * nonzero_mask
        sorted_indices = state_masked.argsort(dim=1, descending=True)
        return indices.gather(1, sorted_indices)
