import torch
import torch.nn as nn

# 模拟 MegatronModule 基类
class MegatronModule(nn.Module):
    def __init__(self):
        super().__init__()

# 模拟 TransformerBlock（简化为一个线性层）
class TransformerBlock(MegatronModule):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.layernorm(x)
        x = self.linear(x)
        return x

# 模拟 GPTModel（由多个 TransformerBlock 组成）
class GPTModel(MegatronModule):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Float16Module 类实现
class Float16Module(MegatronModule):
    def __init__(self, module, fp16_enabled=True):
        super().__init__()
        self.module = module  # 原始模型
        self.fp16_enabled = fp16_enabled
        self.cast_to_fp16()  # 转换为 FP16

    def cast_to_fp16(self):
        if self.fp16_enabled:
            self.module = self.module.half()  # 转换为 FP16

    def forward(self, *args, **kwargs):        
        # 将输入转换为 FP16（如果需要）
        if self.fp16_enabled:
            args = tuple(arg.half() if arg.dtype == torch.float32 else arg for arg in args)
            
            
        return self.module(*args, **kwargs)
    

# 设置参数
hidden_size = 512
num_layers = 4
batch_size = 8
seq_length = 32

# 创建 GPTModel 实例
model = GPTModel(hidden_size=hidden_size, num_layers=num_layers)

# 将模型包装到 Float16Module 中以启用 FP16
fp16_model = Float16Module(model, fp16_enabled=True)

# 移动到 GPU（假设使用 CUDA）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fp16_model = fp16_model.to(device)

# 创建输入数据
input_data = torch.randn(batch_size, seq_length, hidden_size).to(device)

# 前向传播
with torch.cuda.amp.autocast():  # 使用 AMP 确保混合精度
    output = fp16_model(input_data)


# 输出结果
print("Input shape:", input_data.shape)
print("Output shape:", output.shape)
print("Output dtype:", output.dtype)  # 应为 torch.float16

# 检查模型参数的精度
for name, param in fp16_model.named_parameters():
    print(f"Parameter {name} dtype: {param.dtype}")
