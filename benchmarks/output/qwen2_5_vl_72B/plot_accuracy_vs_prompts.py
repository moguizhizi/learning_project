import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# ======= 配置区域 =======
folder_path = "/home/project/learning_project/benchmarks/output/qwen2_5_vl_72B"
hardware_filter = "NVIDIA RTX A6000"
output_image = os.path.join(folder_path, "accuracy_vs_num_prompts.png")
# ========================

# {model_name: {num_prompts (int): accuracy_rate}}
data = defaultdict(dict)

# 遍历所有 JSON 文件
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
                        prompts = int(prompts)  # 确保横坐标是数值类型
                        data[model][prompts] = acc
        except Exception as e:
            print(f"跳过错误文件: {file_name}, 错误: {e}")

# 绘图
plt.figure(figsize=(10, 6))
for model_name, prompt_dict in data.items():
    xs = sorted(prompt_dict.keys())
    ys = [prompt_dict[x] for x in xs]
    plt.plot(xs, ys, marker='o', label=model_name)

    # 标出每个点的完整坐标 (x, y)
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
