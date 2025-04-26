from functools import partial

# 定义一个原始函数，用于计算两数之和
def add(x, y):
    return x + y

# 使用partial创建新函数，固定x为5
add_five = partial(add, 5)

# 调用新函数，等同于调用add(5, 3)
result = add_five(3)
print(result)  # 输出8