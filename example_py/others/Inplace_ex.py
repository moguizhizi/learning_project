import torch
import torch.nn as nn

x = torch.randn(3, 3)
y = x.view(-1)
print(y)
y.add_(1)  # inplace 操作
print(y)
print(x)


m = nn.LeakyReLU(0.1, True)
input = torch.randn(2)
print(input)
output = m(input)

print(input)
print(output)

