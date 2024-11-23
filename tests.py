# Tests for custom components of transformer
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

from network import LinearLayer

class TestLinearLayer(unittest.TestCase):

    def setUp(self):
        self.input_dim = 4
        self.output_dim = 3
        self.layer = LinearLayer(self.input_dim, self.output_dim)

    
    def test_forward_shape(self):
        x = torch.randn(2, self.input_dim)
        output = self.layer(x)
        self.assertEqual(output.shape, (2, self.output_dim), "Output sizes doesn't match in LinearLayer")
    

if __name__ == '__main__':
    unittest.main()