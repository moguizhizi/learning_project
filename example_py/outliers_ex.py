import torch

# 假设这是一个权重向量
weights = torch.tensor([0.1, 0.2, -0.3, 0.5, 0.4, 10.0])  # 注意 10.0 是一个异常值（outlier）

print("原始权重：", weights)

scale = weights.abs().max() / 127
quantized = torch.round(weights / scale).clamp(-127, 127)

print("scale =", scale.item())
print("量化后权重：", quantized)


mean = weights.mean()
std = weights.std()
threshold = mean + 1 * std

print("阈值（mean+1*std）:", threshold.item())

# clip 掉异常值
clipped_weights = weights.clamp(min=-threshold, max=threshold)
print("抑制后权重：", clipped_weights)

# 然后再量化
scale = clipped_weights.abs().max() / 127
print("scale =", scale.item())
quantized = torch.round(clipped_weights / scale).clamp(-127, 127)
print("量化后（抑制）权重：", quantized)
