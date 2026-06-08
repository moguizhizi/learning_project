import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.modules):
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.tranpose(-1,-2))
        scores = scores / (self.embed_dim ** 0.5)
        if mask is not None:
             scores.masked_fill(mask==0, "-inf")

        atten_weight = F.softmax(scores, dim=-1)
        atten_out = torch.matmul(atten_weight, v)
        out = self.out_proj(atten_out)
        return out    
