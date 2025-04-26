from einops import einsum
import torch

A = torch.randn(3,4)
B = torch.randn(4,5)

c = einsum(A, B, "i j, j k -> i k")
print(c.size())

A = torch.randn(3,4)
B = torch.randn(3,4)
c = einsum(A, B, "i j, i j -> i j")
print(c.size())

A = torch.randn(3,4)
c = einsum(A, "i j -> j i")
print(c.size())

A = torch.randn(2, 3, 4)  # 形状 (batch, 3, 4)
B = torch.randn(2, 4, 5)  # 形状 (batch, 4, 5)
c = einsum(A, B, "b i j, b j k -> b i k") 
print(c.size())