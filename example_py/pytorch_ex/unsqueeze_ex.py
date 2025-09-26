import torch

data = torch.tensor([10, 20, 30])   # 1D 向量，shape=(3,)
out = data.unsqueeze(1)             # 在第1维（即“列”方向）插入一个维度

print(out)        # tensor([[10],
                  #         [20],
                  #         [30]])
print(out.shape)  # torch.Size([3, 1])