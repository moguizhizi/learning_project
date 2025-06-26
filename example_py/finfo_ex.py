import torch

# 指定一个浮点类型，例如 torch.float32
dtype = torch.float32

# 使用 torch.finfo 获取该类型的浮点信息
finfo = torch.finfo(dtype)

# 输出各种属性
print(f"Data type: {dtype}")
print(f"Bits: {finfo.bits}")              # 总位数
print(f"Precision (eps): {finfo.eps}")    # 机器精度（1 + eps 是第一个可区分的数）
print(f"Maximum value: {finfo.max}")      # 最大值
print(f"Minimum value: {finfo.min}")      # 最小值
print(f"Tiny value: {finfo.tiny}")        # 最小的正规范数