import torch

x = torch.tensor([1,2,3])
print("size:", x.size())

y = x.repeat(3)
print("size:", y.size())

z = x.repeat(2,1)
print("size:", z.size())

x = torch.tensor([[1,2],[3,4]])
print("size:", x.size())

y = x.repeat(3,1)
print("size:", y.size())

z = x.repeat(3,2)
print("size:", z.size())