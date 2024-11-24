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
    

    def test_forward_values(self):
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        output = self.activation(x)
        print(f'Input x: {x}')
        print(f'Approx GELU output: {output}')

        # Detailed checks:
        for i, (out, inp) in enumerate(zip(output, x)):
            print(f"Index {i}: Output = {out}, Input = {inp}")
            if inp < 0:
                # For negative inputs, output should be closer to zero than the input
                self.assertTrue(out > inp, f"At index {i}, output {out} should be greater than input {inp} for negative x")
            else:
                # For non-negative inputs, output should not exceed the input
                self.assertTrue(out <= inp, f"At index {i}, output {out} should not exceed input {inp} for positive x")


    def test_gradient(self):
        x = torch.randn(5, requires_grad=True)
        output = self.activation(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad, 'Gradients for x should exist.')
        print(f'Gradient of x: {x.grad}')

    
    def test_gelu_comparasion(self):
        x = torch.randn(10)
        output_custom = self.activation(x)
        output_built_in = F.gelu(x)
        self.assertTrue(torch.allclose(output_custom, output_built_in, atol=1e-3), 'Outputs should differ less than 1e-3')
        print(f'Approx GELU: {output_custom}')
        print(f'Build_in GELU: {output_built_in}')
    

    def test_edge_cases(self):
        
        large_value = torch.tensor(1e6, requires_grad=True)
        small_value = torch.tensor(1e-6, requires_grad=True)

        large_output = self.activation(large_value)
        small_output = self.activation(small_value)
        print(f'Large output size: {large_output}')
        self.assertAlmostEqual(large_value.item(), large_output.item(), delta=1e-3, msg='Values should be of similar size.')
        self.assertAlmostEqual(0, small_output.item(), delta=1e-3, msg='Answer should be close to 0')


    def test_tanh(self):

        x = torch.tensor([-1.0, 1.0, -1.0])
        tanh_output = self.activation.tanh(x)
        expected_output = torch.tanh(x)
        print(f'Custom tanh: {tanh_output}')
        print(f'Torch tanh: {expected_output}')
        self.assertTrue(torch.allclose(tanh_output, expected_output, atol=1e-6))
        

if __name__ == '__main__':
    unittest.main()