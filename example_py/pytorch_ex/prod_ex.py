import torch

# 假设一次 forward 里共有 3 条序列，长度分别为 4、5、6
batch_dims = [4,5,6]          # 1-D 列表（或元组）

# 代码中的计算：
batch_size = torch.prod(torch.tensor(batch_dims))

print(batch_size.shape)


print("batch_dims =", batch_dims)
print("torch.tensor(batch_dims) =", torch.tensor(batch_dims))
print("torch.prod(...) =", torch.prod(torch.tensor(batch_dims)))
print("batch_size =", batch_size)