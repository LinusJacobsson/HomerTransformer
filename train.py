import torch
import torch.nn as nn
import network
import tokenizer
import torch.optim as optim
from tqdm import tqdm

with open('shakespeare.txt', 'r') as file:
    text = file.read()

print(text[:100])


tokenizer_ = tokenizer.WordTokenizer()
vocab = tokenizer_.create_vocab(text)
print(len(vocab))
encoded_text = tokenizer_.encode(text)

sequence_length = 10
inputs, outputs = [], []

for i in range(len(encoded_text) - sequence_length):
    inputs.append(encoded_text[i:i + sequence_length])
    outputs.append(encoded_text[i + sequence_length])

inputs = torch.tensor(inputs, dtype=torch.long)
outputs = torch.tensor(outputs, dtype=torch.long)

vocab_size = len(vocab)
d_model = 32
num_heads = 4
d_ff = 32
num_layers = 4
max_len = 5000

model = network.TransformerDecoder(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    max_len=max_len
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

batch_size = 32
epochs = 10

# Create batches
num_batches = len(inputs) // batch_size
input_batches = inputs[:num_batches * batch_size].view(num_batches, batch_size, -1)
output_batches = outputs[:num_batches * batch_size].view(num_batches, batch_size)

for epoch in range(epochs):
    epoch_loss = 0

    # Wrap the batch loop with tqdm
    with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
        for i in range(num_batches):
            # Get the current batch
            input_batch = input_batches[i]  # Shape: (batch_size, sequence_length)
            output_batch = output_batches[i]  # Shape: (batch_size)

            # Forward pass
            optimizer.zero_grad()
            predictions = model(input_batch)  # Shape: (batch_size, sequence_length, vocab_size)
            predictions = predictions[:, -1, :]  # Only the last token prediction matters

            # Compute loss
            loss = criterion(predictions, output_batch)
            epoch_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update tqdm
            pbar.set_postfix({"Batch Loss": loss.item()})
            pbar.update(1)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / num_batches:.4f}")