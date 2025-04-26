import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(5, 3)

    def forward(self, x):
        return self.fc(x)

model = SimpleNet()
# 定义钩子函数
def hook(module, input, output):
    print(f"模块 {module.__class__.__name__} 的输入形状: {input[0].shape}")
    print(f"模块 {module.__class__.__name__} 的输出形状: {output.shape}")

handle = model.fc.register_forward_hook(hook)
x = torch.randn(2, 5)
output = model(x)
handle.remove()  # 用完后移除钩子