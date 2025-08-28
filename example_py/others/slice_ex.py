# 创建一个切片对象
my_slice = slice(1, 5, 2)

# 使用切片对象对列表进行切片
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
result = numbers[my_slice]

print(result)  # 输出: [1, 3, 5]
print(my_slice)

# 创建切片对象
s = slice(2, 7)

# 访问切片的属性
print(s.start)  # 输出: 2
print(s.stop)   # 输出: 7
print(s.step)   # 输出: None（默认步长为 1）

# 应用到字符串
text = "Hello, World!"
result = text[s]
print(result)  # 输出: llo, Wo

import torch

# 创建一个 3×4 的张量
tensor = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])

# 创建切片对象
row_slice = slice(0, 2)  # 取前两行
col_slice = slice(1, 3)  # 取第 2 和第 3 列

# 使用切片对象
result = tensor[row_slice, col_slice]
print(result)

# 创建一个列表
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 使用负索引和步长
my_slice = slice(-5, -1, 1)

result = numbers[my_slice]
print(result)  # 输出: [5, 6, 7, 8]