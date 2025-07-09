import os
import json
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

# ========= 解析命令行参数 =========
parser = argparse.ArgumentParser(description="绘制 accuracy vs num_prompts 曲线")
parser.add_argument("--hardware_filter", type=str, required=True, help="筛选的 GPU 硬件名称，例如 'NVIDIA RTX A6000'")
args = parser.parse_args()
hardware_filter = args.hardware_filter
# =================================

# 固定路径
folder_path = "/home/project/learning_project/benchmarks/output/qwen2_5_vl_72B"
output_image = os.path.join(folder_path, f"{hardware_filter}_accuracy_vs_num_prompts.png")

# {model_name: {num_prompts: accuracy_rate}}
data = defaultdict(dict)

# 遍历 JSON 文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".json") and "openai-chat-infqps" in file_name:
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, "r") as f:
                content = json.load(f)
                if content.get("hardware") == hardware_filter:
                    model = content.get("model_name")
                    prompts = content.get("num_prompts")
                    acc = content.get("accuracy_rate")
                    if model and prompts is not None and acc is not None:
                        prompts = int(prompts)
                        data[model][prompts] = acc
        except Exception as e:
            print(f"跳过错误文件: {file_name}, 错误: {e}")

# 绘图
plt.figure(figsize=(10, 6))
for model_name, prompt_dict in data.items():
    xs = sorted(prompt_dict.keys())
    ys = [prompt_dict[x] for x in xs]
    plt.plot(xs, ys, marker='o', label=model_name)

    # 标出每个点的坐标
    for x, y in zip(xs, ys):
        plt.text(x, y, f"({x}, {y:.1f})", fontsize=9, ha='right', va='bottom', color='black')

plt.title(f"Accuracy vs Num Prompts (hardware = {hardware_filter})")
plt.xlabel("num_prompts")
plt.ylabel("accuracy_rate (%)")
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.savefig(output_image)
print(f"图像已保存为: {output_image}")
