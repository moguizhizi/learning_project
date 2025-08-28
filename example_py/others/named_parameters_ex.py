from transformers import Qwen2Model, Qwen2Config
import torch

# 1) 任选一种方式拿到模型
# 1-A) 直接下载并缓存权重（需要联网）
model = Qwen2Model.from_pretrained("/home/llm_model/Qwen/Qwen3-0___6B")

# 1-B) 或者本地随机初始化一个尺寸最小的 Qwen2 做演示
# config = Qwen2Config(
#     hidden_size=256,
#     intermediate_size=512,
#     num_hidden_layers=2,
#     num_attention_heads=8,
#     vocab_size=32000
# )
# model = Qwen2Model(config)

model.eval()

# 2) 遍历所有参数（不去重，与调用 model.named_parameters() 等价，但显式写 remove_duplicate=False）
for name, param in model.named_parameters(remove_duplicate=False):
    # 打印名字、形状、是否需要梯度
    print(f"{name:60} {str(param.shape):20} requires_grad={param.requires_grad}")

# 3) 如果你想统计总参数量
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\nTotal params:", total)
print("Trainable params:", trainable)
