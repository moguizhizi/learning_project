# import torch
# from math import sqrt
# from torch.nn import functional as F
# import torch.nn as nn


# def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, query_mask=None, key_mask=None, mask=None):
#     hide_dim = query.size()
#     scores = torch.bmm(query, key.transpose(1, 2))/sqrt(hide_dim)
#     if query_mask is not None and key_mask is not None:
#         mask = torch.bmm(query.unsqueeze(-1), key.unsqueeze(1))
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, float("-inf"))

#     weight = F.softmax(scores, -1)
#     return torch.bmm(weight, value)


# class AttentionHead(nn.Module):
#     def __init__(self, embed_dim, hide_dim):
#         super().__init__()
#         self.q = nn.Linear(embed_dim, hide_dim)
#         self.k = nn.Linear(embed_dim, hide_dim)
#         self.v = nn.Linear(embed_dim, hide_dim)

#     def forward(self, query, key, value, query_mask, key_mask, mask):
#         query = self.q(query)
#         key = self.k(key)
#         value = self.v(value)
#         return scaled_dot_product_attention(query, key, value, query_mask, key_mask, mask)


# class MultiHeadAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         embed_dim = config.hidden_size
#         num_heads = config.num_attention_heads
#         head_dim = embed_dim // num_heads
#         self.heads = nn.ModuleList(
#             [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
#         self.output = nn.Linear(embed_dim, embed_dim)

#     def forward(self, query, key, value, query_mask, key_mask, mask):
#         x = torch.cat([h(query, key, value, query_mask, key_mask, mask)
#                       for h in self.heads], dim=-1)
#         return self.output(x)
