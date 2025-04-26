import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

torch.manual_seed(42)


class TestModel(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, 10)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.linear(x)
        x = self.lm_head(x)
        return x


# set a reasonably large hidden size to illustrate the small fraction of
# params needed to be added for LoRA
hidden_size = 1024
model = TestModel(hidden_size)

# dummy inputs
input_ids = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]])

# toy example of a detokenizer.
# The vocabulary only consists of 10 words (different colors)
detokenizer = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "indigo",
    "violet",
    "magenta",
    "marigold",
    "chartreuse",
]

# this is the same generation step as we saw in lesson 2 (batching)
def generate_token(model, **kwargs):
    with torch.no_grad():
        logits = model(**kwargs)
    last_logits = logits[:, -1, :]
    next_token_ids = last_logits.argmax(dim=1)

    return [detokenizer[token_id] for token_id in next_token_ids]

# generate one token
next_token = generate_token(model, input_ids=input_ids)[0]
print(next_token)

# dummy input tensor
# shape: (batch_size, sequence_length, hidden_size)
X = torch.randn(1, 8, 1024)

# LoRA A and B tensors
# A has shape (hidden_size, rank)
# B has shape (rank, hidden_size)
lora_a = torch.randn(1024, 2)
lora_b = torch.randn(2, 1024)

W = model.linear.weight
W2 = lora_a @ lora_b

# Compare number of elements of A and B with number of elements of W
# W here has shape (hidden_size, hidden_size)
lora_numel = lora_a.numel() + lora_b.numel()
base_numel = W.numel()
print("|A+B| / |W|:", lora_numel / base_numel)

# compute the output of X @ W (the original linear layer)
base_output = model.linear(X)

# compute the output of X @ A @ B (the added lora adapter)
lora_output = X @ lora_a @ lora_b

# sum them together
total_output = base_output + lora_output

# output should have the same shape as the original output:
# (batch_size, sequence_length, hidden_size)
print(total_output.shape)

class LoraLayer(torch.nn.Module):
    def __init__(self, base_layer, r):
        super().__init__()
        self.base_layer = base_layer
        
        d_in, d_out = self.base_layer.weight.shape
        self.lora_a = torch.randn(d_in, r)
        self.lora_b = torch.randn(r, d_out) 
        
    def forward(self, x):
        y1 = self.base_layer(x)
        y2 = x @ self.lora_a @ self.lora_b
        return y1 + y2
    
# wrap the linear layer of our toy model, use rank 2
lora_layer = LoraLayer(model.linear, 2)
lora_layer(X).shape

model.linear = lora_layer

print(model)

next_token = generate_token(model, input_ids=input_ids)
print(next_token[0])