from torch import nn as nn
import torch
from torch.nn import functional as F
from math import sqrt


class DeepSpeedInferenceAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int):
        super().__init__()

        self._d_model = d_model
        self._num_heads = num_heads
        self._heads_dim = d_model // num_heads
        self._max_seq_len = max_seq_len

        self._key_cache = None
        self._value_cache = None
        self._current_seq_len = 0

    def _accolate_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        cache_size = (batch_size, self._num_heads,
                      self._max_seq_len, self._heads_dim)
        self._key_cache = torch.zeros(cache_size, device=device, dtype=dtype)
        self._value_cache = torch.zeros(cache_size, device=device, dtype=dtype)
        self._current_seq_len = 0

    def _update_cache(self, key: torch.Tensor, value: torch.Tensor):
        batch_size, seq_len, _ = key.shape()

        key = key.view(batch_size, seq_len, self._num_heads,
                       self._heads_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self._num_heads,
                           self._heads_dim).transpose(1, 2)

        self._key_cache[:, :,
                        self._current_seq_len:self._current_seq_len+seq_len, :] = key
        self._value_cache[:, :,
                          self._current_seq_len:self._current_seq_len+seq_len, :] = value

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, use_cache):
        batch_size, seq_len, _ = query.shape()
        device = query.device()
        dtype = query.dtype()

        if self._key_cache is None and use_cache:
            self._accolate_cache(batch_size, device=device, dtype=dtype)

        if use_cache:
            self._update_cache(key, value)

        if use_cache:
            key = self._key_cache[:, :, self._current_seq_len, :]
            value = self._value_cache[:, :, self._current_seq_len, :]
        else:
            key = key.view(batch_size, seq_len, self._num_heads,
                           self._heads_dim).transpose(1, 2)
            value = value.view(batch_size, seq_len,
                               self._num_heads, self._heads_dim).transpose(1, 2)

        query = query.view(batch_size, seq_len,
                           self._num_heads, self._heads_dim)

        scores = torch.bmm(query, key.transpose(-2, -1)) / \
            sqrt(self._heads_dim)
        weight = F.softmax(scores, dim=-1)
        value = torch.bmm(weight, value).transpose(
            1, 2).contiguous.view(batch_size, seq_len, self._d_model)

        return value
