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
        return self.slope * torch.exp(-self.slope * x) / ((torch.exp(-self.slope * x) + 1) ** 2)

class FastSigmoid(SurrogateThresholdFunction):
    def __init__(self, slope=25.0):
        self.slope = slope
    
    def backward(self, x: torch.Tensor):
        return 1.0 / (1.0 + self.slope * torch.abs(x))**2
    
class Atan(SurrogateThresholdFunction):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def backward(self, x: torch.Tensor):
        return 1 / (torch.pi * (1 + (torch.pi * x * (self.alpha/2))**2))
