import torch

x = torch.arange(10)        # tensor([0, 1, 2, ..., 9])
y = x.narrow(0, 2, 5)       # tensor([2, 3, 4, 5, 6])
y[0] = 100
print(x)                    # tensor([  0,   1, 100,   3,   4,   5,   6,   7,   8,   9])
