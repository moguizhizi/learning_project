from torch import nn as nn
from torch.nn import functional as F
import torch
import math


class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self._d_model = d_model
        self._num_heads = num_heads
        self._head_dim = d_model // num_heads
        self._kv_cache = None

    def forward(self, q, k, v):
        if self._kv_cache is None:
            self._kv_cache = (k, v)
        else:
            self._kv_cache = (torch.cat([self._kv_cache[0], k], dim=1), torch.cat(
                [self._kv_cache[1], v], dim=1))
        
        k_cache, v_cache = self._kv_cache
        scores = torch.bmm(q, k_cache.transpose(1, 2)) / math.sqrt(self._head_dim)
        weight = F.softmax(scores, dim=-1)
        
        return torch.bmm(weight, v_cache)
    

d_model = 512
num_heads = 8 

att = Attention(512, 8)
q = torch.rand(1, 1, d_model)
k = torch.rand(1, 10, d_model)
v = torch.rand(1, 10, d_model)
output = att(q, k, v)
print(output.size())

k = torch.rand(1, 1, d_model)
v = torch.rand(1, 1, d_model)
output = att(q, k, v)
print(output.size())

