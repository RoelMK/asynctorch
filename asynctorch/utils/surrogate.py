from abc import ABC, abstractmethod
import torch

class SurrogateThresholdFunction(ABC):
    def forward(self, x: torch.Tensor):
        return (x > 0).float()

    @abstractmethod
    def backward(self, x):
        pass

class Sigmoid(SurrogateThresholdFunction):
    def __init__(self, slope=1.0):
        self.slope = slope

    def backward(self, x: torch.Tensor):
        if x.is_sparse:
            # Apply the operation only on the non-zero elements (values)
            values = x._values()
            exp_term = torch.exp(-self.slope * values)
            new_values = self.slope * exp_term / ((exp_term + 1) ** 2)
            # Return a new sparse tensor with the updated values
            return torch.sparse_coo_tensor(x._indices(), new_values, x.shape)
        else:
            return self.slope * torch.exp(-self.slope * x) / ((torch.exp(-self.slope * x) + 1) ** 2)

class FastSigmoid(SurrogateThresholdFunction):
    def __init__(self, slope=25.0):
        self.slope = slope
    
    def backward(self, x: torch.Tensor):
        if x.is_sparse:
            # Apply the operation only on the non-zero elements (values)
            values = x._values()
            new_values = 1.0 / (1.0 + self.slope * torch.abs(values)) ** 2
            # Return a new sparse tensor with the updated values
            return torch.sparse_coo_tensor(x._indices(), new_values, x.shape)
        else:
            return 1.0 / (1.0 + self.slope * torch.abs(x)) ** 2
    
class ATan(SurrogateThresholdFunction):
    def __init__(self, alpha=2.0):
        self.alpha = alpha

    def backward(self, x: torch.Tensor):
        if x.is_sparse:
            # Apply the operation only on the non-zero elements (values)
            values = x._values()
            new_values = 1 / (torch.pi * (1 + (torch.pi * values * (self.alpha / 2)) ** 2))
            # Return a new sparse tensor with the updated values
            return torch.sparse_coo_tensor(x._indices(), new_values, x.shape)
        else:
            return 1 / (torch.pi * (1 + (torch.pi * x * (self.alpha / 2)) ** 2))
