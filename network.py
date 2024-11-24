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


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = LinearLayer(d_model, d_ff)
        self.activation1 = ApproxGELU()
        self.linear2 = LinearLayer(d_ff, d_model)


    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    
    def forward(self, x):
        x_mean = torch.mean(x, -1)
        x_variance = torch.var(x, -1)
        normal_term = torch.sqrt(x_variance**2 + self.epsilon) * self.gamma
        x = (x - x_mean) * normal_term + self.beta
        return x

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
    


