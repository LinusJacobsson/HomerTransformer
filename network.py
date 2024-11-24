# Network classes 
import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))


    def forward(self, x):
        if x.shape[0] == 0:
            raise RuntimeError("Tensor has no data (empty batch dimension)")
        return torch.matmul(x, self.weights) + self.bias



class ApproxGELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def tanh(self, x):
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
    

    def forward(self, x):
        # Approximate according to: https://arxiv.org/pdf/1606.08415
        # Corrected GELU approximation formula
        term = torch.sqrt(torch.tensor(2.0) / torch.pi) * (x + 0.044715 * x**3)
        term = torch.clamp(term, -50, 50)
        return 0.5 * x * (1 + self.tanh(term)) 
    


