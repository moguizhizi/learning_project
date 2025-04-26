import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 10 * 10, 128)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 10 * 10)
        x = torch.relu(self.fc1(x))
        return x

model = MyModel()

# 使用 named_children() 遍历
print("named_children() 结果：")
for name, child in model.named_children():
    print(f"名称: {name}, 模块类型: {type(child)}")

# 使用 named_modules() 遍历
print("\nnamed_modules() 结果：")
for name, module in model.named_modules():
    print(f"名称: {name}, 模块类型: {type(module)}")