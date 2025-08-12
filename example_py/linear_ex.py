import torch

# 假设 fc.weight 形状 [3, 4]
fc = torch.nn.Linear(4, 3)
# fc.weight.data = torch.tensor([[1.0, 2.0, 3.0, 4.0],
#                                [5.0, 6.0, 7.0, 8.0],
#                                [9.0,10.0,11.0,12.0]])

# # 4 个输入通道的缩放因子
# scales = torch.tensor([0.1, 0.2, 0.3, 0.4])

# with torch.no_grad():
#     fc.weight.mul_(scales.view(1, -1))

print(fc.weight.size())