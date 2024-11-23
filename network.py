# Network classes 
import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weights = nn.Parameter(input_dim, output_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        return x @ self.weight + self.bias
