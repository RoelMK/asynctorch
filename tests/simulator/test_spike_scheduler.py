from asynctorch.nn.neuron.lif_state import LIFState
from asynctorch.simulator.spike_scheduler import MomentumSpikeScheduler, RandomSpikeScheduler
import torch
from snntorch import surrogate

def test_random_spike_scheduler_single_spike():
    random_scheduler = RandomSpikeScheduler(None)
    out_spikes = torch.asarray([[1,0,0], [0,1,0], [0,0,1]])
    indices = torch.asarray([[0,1,2], [0,1,2], [0,1,2]])
    sorted_indices = random_scheduler.sort(out_spikes, indices)
    assert sorted_indices.shape == (3,3)
    assert sorted_indices[0, 0] == 0
    assert sorted_indices[1, 0] == 1
    assert sorted_indices[2, 0] == 2

def test_random_spike_scheduler_double_spike():
    random_scheduler = RandomSpikeScheduler(None)
    out_spikes = torch.asarray([[1,1,0], [0,1,1], [1,0,1]])
    indices = torch.asarray([[0,1,2], [0,1,2], [0,1,2]])
    sorted_indices = random_scheduler.sort(out_spikes, indices)
    assert sorted_indices.shape == (3,3)
    assert sorted_indices[0, 0] == 0 or sorted_indices[0, 0] == 1
    assert sorted_indices[1, 0] == 1 or sorted_indices[1, 0] == 2
    assert sorted_indices[2, 0] == 2 or sorted_indices[2, 0] == 0


def test_momentum_spike_scheduler():
    lif_state = LIFState(
        neurons_per_layer=[2],
        tau_m=torch.tensor([1.0, 1.0]),
        membrane_threshold=torch.tensor([0.2, 0.2]),
        sync_threshold=torch.tensor([0.0, 0.0]),
        spike_grad=surrogate.atan(),
        sync_grad=surrogate.atan(),
        device=torch.device("cpu")
    )
    lif_state._init_state(3)
    new_spikes = lif_state(torch.tensor([[0.5, 0.6], [0.1, 0.3], [0.9, 0.8]]), torch.tensor([[1,1], [1,1], [1,1]]))
    scheduler = MomentumSpikeScheduler(lif_state)
    sorted_indices = scheduler.sort(new_spikes, torch.tensor([[0,1], [0,1], [0,1]]))
    assert sorted_indices.shape == (3,2)
    assert sorted_indices[0, 0] == 1
    assert sorted_indices[1, 0] == 1
    assert sorted_indices[2, 0] == 0
