# Tests for custom components of transformer
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

from network import LinearLayer, ApproxGELU

class TestLinearLayer(unittest.TestCase):

    def setUp(self):
        self.input_dim = 4
        self.output_dim = 3
        self.layer = LinearLayer(self.input_dim, self.output_dim)

    
    def test_forward_shape(self):
        x = torch.randn(2, self.input_dim)
        output = self.layer(x)
        self.assertEqual(output.shape, (2, self.output_dim), "Output sizes doesn't match in LinearLayer")


    def test_paramters(self):
        # Check that parameters work correctly
        parameters = list(self.layer.parameters())
        self.assertEqual(len(parameters), 2, 'LinearLayer should have 2 sets of parameters: weights and biases')
        self.assertEqual(parameters[0].shape, (self.input_dim, self.output_dim), "Weights should be (input_dim, output_dim)")
        self.assertEqual(parameters[1].shape, (self.output_dim, ), 'Biases should have shape (output_dim, )') 

    
    def test_gradient(self):
        x = torch.randn(2, self.input_dim, requires_grad=True)
        output = self.layer(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad, 'Gradients should exist.')
        self.assertIsNotNone(self.layer.weights.grad, 'Gradients should exist.')
        self.assertIsNotNone(self.layer.bias.grad, 'Gradients should exist.')


    def test_empty_tensor(self):
        x = torch.empty(0, self.input_dim)
        with self.assertRaises(RuntimeError):
            self.layer(x)



class TestApproximateGELU(unittest.TestCase):

    def setUp(self):
        self.activation = ApproxGELU()


    def test_forward_shape(self):
        # Test that shapes agree
        x = torch.randn(10, 5)
        output = self.activation(x)
        self.assertEqual(x.shape, output.shape, 'Output shape must match the input shape')
    

if __name__ == '__main__':
    unittest.main()