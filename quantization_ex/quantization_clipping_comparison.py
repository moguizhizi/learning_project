import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def quantize(x, scale, clip_min=-128, clip_max=127):
    """模拟量化和反量化过程"""
    q = torch.clamp((x / scale).round(), clip_min, clip_max)
    return q * scale

def compute_metrics(original, quantized):
    """计算表达精度指标"""
    mse = torch.mean((original - quantized) ** 2).item()
    mae = torch.mean(torch.abs(original - quantized)).item()
    max_err = torch.max(torch.abs(original - quantized)).item()
    cosine = F.cosine_similarity(original.unsqueeze(0), quantized.unsqueeze(0)).item()
    return mse, mae, max_err, cosine

# 模拟激活值（含异常值）
x = torch.tensor([-20.0, -2, -1, 0, 1, 2, 20.0])

# ========= 方式一：不裁剪 =========
scale1 = (x.max() - x.min()) / 255
x_q1 = quantize(x, scale1)
mse1, mae1, max_err1, cos1 = compute_metrics(x, x_q1)

# ========= 方式二：裁剪异常值 =========
clip_val = 3
x_clipped = torch.clamp(x, -clip_val, clip_val)
scale2 = (x_clipped.max() - x_clipped.min()) / 255
x_q2 = quantize(x_clipped, scale2)
mse2, mae2, max_err2, cos2 = compute_metrics(x_clipped, x_q2)

# ========= 输出指标 =========
print("【不裁剪】")
print("量化结果:", x_q1)
print(f"MSE={mse1:.4f}, MAE={mae1:.4f}, MaxErr={max_err1:.4f}, Cosine={cos1:.4f}\n")

print("【裁剪到 [-3, 3]】")
print("量化结果:", x_q2)
print(f"MSE={mse2:.4f}, MAE={mae2:.4f}, MaxErr={max_err2:.4f}, Cosine={cos2:.4f}")

# ========= 可视化 =========
plt.figure(figsize=(10, 5))
plt.plot(x.numpy(), label="Original", marker='o')
plt.plot(x_q1.numpy(), label="Quantized (No Clipping)", marker='x')
plt.plot(x_q2.numpy(), label="Quantized (Clipped)", marker='^')
plt.title("Quantization Comparison")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.tight_layout()

# 保存图像到文件
output_path = "/home/project/learning_project/quantization_ex/quantization_comparison.png"
plt.savefig(output_path)