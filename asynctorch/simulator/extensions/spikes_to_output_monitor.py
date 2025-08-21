from typing import List, Tuple
import torch

from asynctorch.simulator.extensions.forward_step_extension import ForwardStepMonitor

class SpikesToOutputMonitor(ForwardStepMonitor):
    step_classifications: List[Tuple[torch.Tensor, torch.Tensor, int]]
    
    def __init__(self, n_neurons, n_outputs):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self._reset_state()

    def _reset_state(self):
        self.step_classifications = []
        self.i_forward = 0

    def monitor_spike(self, new_spikes_current_step: torch.Tensor, all_spikes_current_forward: torch.Tensor, all_spikes_previous_forwards: torch.Tensor, is_input: bool):
        if is_input:
            self.i_forward += 1
        else:
            s_out_class_layer = new_spikes_current_step[:, self.n_neurons-self.n_outputs:]
            if s_out_class_layer.sum() > 0:
                self.step_classifications.append(((all_spikes_current_forward + new_spikes_current_step).sum(dim=1), s_out_class_layer, self.i_forward))

    def compute_latencies(self, target_classes: torch.Tensor) -> Tuple[List[int], List[int]]:
        batch_size = target_classes.shape[0]

        # Find the first spike time for each forward step and each batch element
        n_spikes_to_correct_classification = {} # Dict of forward step index to list of spike times for each batch element
        n_spikes_to_any_classification = {}
        for n_spikes, s_out, i_forward in self.step_classifications:
            # If the forward step index is not in the dict, add it
            if n_spikes_to_any_classification.get(i_forward) is None:
                n_spikes_to_any_classification[i_forward] = [-1 for _ in range(batch_size)]
            if n_spikes_to_correct_classification.get(i_forward) is None:
                n_spikes_to_correct_classification[i_forward] = [-1 for _ in range(batch_size)]
            # Update the spike times for each batch element
            for j in range(batch_size):
                n_spikes_batch = n_spikes[j].item()
                if s_out[j].sum() > 0 and (n_spikes_to_any_classification[i_forward][j] == -1 or n_spikes_batch < n_spikes_to_any_classification[i_forward][j]):
                    n_spikes_to_any_classification[i_forward][j] = n_spikes_batch
                if s_out[j, target_classes[j]] > 0 and (n_spikes_to_correct_classification[i_forward][j] == -1 or n_spikes_batch < n_spikes_to_correct_classification[i_forward][j]):
                    n_spikes_to_correct_classification[i_forward][j] = n_spikes_batch

        # Flatten the lists
        n_spikes_to_any_classification_flat = [n for sublist in n_spikes_to_any_classification.values() for n in sublist]
        n_spikes_to_correct_classification_flat = [n for sublist in n_spikes_to_correct_classification.values() for n in sublist]
        # Filter out the -1s
        n_spikes_to_any_classification_flat = [n for n in n_spikes_to_any_classification_flat if n != -1]
        n_spikes_to_correct_classification_flat = [n for n in n_spikes_to_correct_classification_flat if n != -1]

        return n_spikes_to_any_classification_flat, n_spikes_to_correct_classification_flat