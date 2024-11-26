# Tests for custom components of transformer
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

from network import LinearLayer, ApproxGELU, FeedForwardBlock, LayerNorm, AddAndNorm, EmbeddingLayer

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
        


class TestFeedForwardBlock(unittest.TestCase):

    def setUp(self):
        self.d_model = 16
        self.d_ff = 64
        self.block = FeedForwardBlock(d_model=self.d_model, d_ff=self.d_ff)
    

    def test_forward_shape(self):

        batch_size, context_length = 8, 10
        x = torch.randn(batch_size, context_length, self.d_model)
        output = self.block(x)
        self.assertEqual(output.shape, x.shape, 'Shape should be the same after forward pass')

    
    def test_forward_values(self):

        x  =torch.randn(2, 5, self.d_model) # Batch size = 2, context length = 5
        output = self.block(x)
        self.assertTrue(torch.all(torch.isfinite(output)), 'Output contain Nan or None values')


    def test_gradients(self):
        x = torch.randn(4, 6, self.d_model, requires_grad=True) # batch size = 4, context length = 6
        output = self.block(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad, 'Gradient for x should exist.')
        for params in self.block.parameters():
            self.assertIsNotNone(params.grad, 'Gradients should exist for all parameters.')


    def test_empty_input(self):
        x = torch.empty(0, 0, self.d_model) # empty tensor
        with self.assertRaises(RuntimeError):
            self.block(x)
    


class TestLayerNorm(unittest.TestCase):
    
    def setUp(self):
        self.d_model = 16
        self.layer_norm = LayerNorm(d_model=self.d_model)

    
    def test_output_shape(self):
        batch_size, context_length = 8, 10
        x = torch.randn(batch_size, context_length, self.d_model)
        output = self.layer_norm(x)
        self.assertEqual(x.shape, output.shape, 'Shape must be same after forward pass.')


    def test_output_values(self):
        batch_size, context_length = 8, 10
        x = torch.randn(batch_size, context_length, self.d_model) # Shape (8, 10, 16)
        output = self.layer_norm(x)

        output_mean = output.mean(dim=-1)
        output_var = output.var(dim=-1)
        #print(f'Output mean: {output_mean}')
        #print(f'Output variance: {output_var}')
        self.assertTrue(torch.allclose(output_mean, torch.zeros_like(output_mean), atol=1e-6), 'Mean should be close to zero.')
        self.assertTrue(torch.allclose(output_var, torch.ones_like(output_var), atol=1e-6), 'Variance should be close to 1')


    def test_learnable_parameters(self):
        self.assertTrue(self.layer_norm.beta.requires_grad, 'Beta should be a learnable parameter.')
        self.assertTrue(self.layer_norm.gamma.requires_grad, 'Gamma should a learnable parameter')
        self.assertTrue(torch.allclose(self.layer_norm.gamma, torch.ones_like(self.layer_norm.gamma), atol=1e-9), 'Gamma should be initialized to one.')
        self.assertTrue(torch.allclose(self.layer_norm.beta, torch.zeros_like(self.layer_norm.beta), atol=1e-9), 'Beta should be initialized to zero.')

    
    def test_gradients(self):
        batch_size, context_length = 5, 5
        x = torch.randn(batch_size, context_length, self.d_model, requires_grad=True) # Shape (5, 5, 16)
        output = self.layer_norm(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad, 'Gradient for x should exist.')
        self.assertIsNotNone(self.layer_norm.gamma.grad, 'Gradient for gamma should exist.')
        self.assertIsNotNone(self.layer_norm.beta.grad, 'Gradient for beta should exist.')


    def test_empty_tensor(self):
        x = torch.empty(0, 0, self.d_model) # Empty tensor
        with self.assertRaises(RuntimeError):
            self.layer_norm(x)


    def test_single_element(self):
        x = torch.tensor([5.0])
        layer_norm = LayerNorm(1)
        output = layer_norm(x)
        print(f'Output: {output}')
        self.assertTrue(torch.allclose(output, torch.zeros_like(output), atol=1e-10), 'Scalar should be normalized to 0.')


class TestAddAndNorm(unittest.TestCase):

    def setUp(self):
        self.d_model = 16
        self.layer = AddAndNorm(self.d_model)
    

    def test_output_shape(self):
        batch_size, context_length = 10, 8
        x = torch.randn(batch_size, context_length, self.d_model)
        sublayer_output = torch.randn(batch_size, context_length, self.d_model)
        output = self.layer(x, sublayer_output)
        self.assertEqual(x.shape, output.shape, 'Input shape and output shape must match.')


    def test_gradient_flow(self):
        x = torch.randn(4, 5, self.d_model, requires_grad=True)
        sublayer_output = torch.randn(4, 5, self.d_model, requires_grad=True)
        output = self.layer(x, sublayer_output)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad, "Gradients for x should exist")
        self.assertIsNotNone(sublayer_output.grad, "Gradients for sublayer_output should exist")


    def test_zero_input(self):
        x = torch.zeros(4, 5, self.d_model)
        sublayer_output = torch.zeros(4, 5, self.d_model)
        output = self.layer(x, sublayer_output)
        self.assertTrue(torch.allclose(output, torch.zeros_like(output), atol=1e-6), "Output should be zero when inputs are zero")



    def test_single_element_tensor(self):
        add_and_norm = AddAndNorm(d_model=1)
        x = torch.tensor([[5.0]])
        sublayer_output = torch.tensor([[3.0]])
        output = add_and_norm(x, sublayer_output)
        print(f'Output mean: {output.mean()}')
        print(f'Output var: {output.var()}')
        self.assertTrue(torch.allclose(output.mean(), torch.tensor(0.0), atol=1e-6), "Mean should be 0 after normalization")
        self.assertTrue(output.var(unbiased=False).isnan() or output.var(unbiased=False) == 0, "Variance should be zero for scalar input.")


class TestEmbeddingLayer(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 50
        self.embedding_dim = 100
        self.batch_size = 32
        self.sequence_length = 10
        self.layer = EmbeddingLayer(self.vocab_size, self.embedding_dim)


    
    def test_output_shape(self):
        tokens = torch.randint(0, self.vocab_size, (self.batch_size, self.sequence_length), dtype=torch.long)
        output = self.layer(tokens)
        self.assertEqual(output.shape, (self.batch_size, self.sequence_length, self.embedding_dim), 'The dimensions are not correct.')

    
    def test_gradients(self):
        tokens = torch.randint(0, self.vocab_size, (self.batch_size, self.sequence_length), dtype=torch.long)
        output = self.layer(tokens)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(self.layer.embeddings.grad, 'Gradients should exist.')


    def test_out_of_vocab(self):
        tokens = torch.tensor([[0, 1, self.vocab_size]])
        with self.assertRaises(IndexError):
            self.layer(tokens)


    def test_single_token(self):
        token = torch.tensor([5], dtype=torch.long)
        output = self.layer(token)
        self.assertEqual(output.shape, (1, self.embedding_dim), 'Should be a single row with embedding_dim integers.')

if __name__ == '__main__':
    loader = unittest.TestLoader()

    # Load specific test methods
    tests = loader.loadTestsFromNames([
        'tests.TestEmbeddingLayer'
    ])

    # Run the selected tests
    runner = unittest.TextTestRunner()
    runner.run(tests)