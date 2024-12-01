# Network classes 
import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.bias = None

    def forward(self, x):
        if x.shape[0] == 0:
            raise RuntimeError("Tensor has no data (empty batch dimension)")
        
        if self.bias is not None:
            return torch.matmul(x, self.weights) + self.bias
        else:
            return torch.matmul(x, self.weights)

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
        if x.shape[0] == 0:
            raise RuntimeError("Tensor has no data (empty batch dimension)")
        # Compute mean and variance along the last dimension
        x_mean = x.mean(dim=-1, keepdim=True)

        if x.size(-1) == 1:
            # Return normalized value directly (gamma * 0 + beta)
            return self.gamma * 0 + self.beta
        
        x_var = x.var(dim=-1, keepdim=True)
        # Normalize
        x_normalized = (x - x_mean) / torch.sqrt(x_var + self.epsilon)
        # Scale and shift
        return self.gamma * x_normalized + self.beta


class AddAndNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = LayerNorm(d_model=d_model)


    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.01)

    
    def forward(self, tokens):
        return self.embeddings[tokens]

    

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))


    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, value)
        return output, attention_weights
    


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
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, num_heads, use_bias=False):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.query_proj = LinearLayer(self.d_model, self.d_model, use_bias=use_bias)
        self.key_proj = LinearLayer(self.d_model, self.d_model, use_bias=use_bias)
        self.value_proj = LinearLayer(self.d_model, self.d_model, use_bias=use_bias)

        # Output
        self.output_proj = LinearLayer(self.d_model, self.d_model, use_bias=use_bias)

        self.attention = ScaledDotProductAttention(self.d_k)

    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size[0]

        # Projections
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) # Shape: (batch_size, seq_len, num_heads, d_k)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attention_output, attention_weights = self.attention(query, key, value, mask) # Shape: (batch_size, num_heads, seq_len, seq_len)

        # Concatenate and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k) # Shape: (batch_size, seq_len, d_model)
        output = self.output_proj(attention_output)

        return output, attention_weights


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attentiom = MultiHeadedAttention(d_model=d_model, num_heads=num_heads)
        self.add_and_norm1 = AddAndNorm(d_model=d_model)
        self.feed_forward = FeedForwardBlock(d_model=d_model, d_ff=d_ff)
        self.add_and_norm2 = AddAndNorm(d_model=d_model)

    
    def forward(self, x, mask=None):
        attention_output = self.attentiom(x)
        norm1 = self.add_and_norm1(x, attention_output)
        feed_forward_output = self.feed_forward(norm1)
        normed_output = self.add_and_norm2(x, feed_forward_output)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5_000):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size=vocab_size, embedding_dim=d_model)
        self.positionl_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.layers = nn.ModuleList([DecoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff) for _ in range(num_layers)])
        self.output = LinearLayer(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)
            
        return self.output(x)