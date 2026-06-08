import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.modules):
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, q, k,v, mask=None):
        batch_size = q.size(0)

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(batch_size, -1, self.num_heads, self.d_dim)
        k = k.view(batch_size, -1, self.num_heads, self.d_dim)
        v = v.view(batch_size, -1, self.num_heads, self.d_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1,-2))
        scores / (self.d_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask==0, float("-inf"))

        atten_weight = F.softmax(scores, dim=-1)
        atten_out = torch.matmul(atten_weight, v)
        atten_out = atten_out.transpose(1,2).contiguous()
        atten_out = atten_out.view(batch_size, -1, self.d_model)
        out = self.out_proj(atten_out)
        return out