from datasets import load_dataset
import os
import json

# 自定义序列化器，处理 bytes 对象
def custom_serializer(obj):
    if isinstance(obj, bytes):  # 假设 bytes 是 image 字段
        # 保存 bytes 到文件
        os.makedirs('images/visionArena_chat', exist_ok=True)
        filename = f'images/visionArena_chat/image_{i}.png'
        with open(filename, 'wb') as bin_file:
            bin_file.write(obj)
        return filename  # 返回文件路径
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# 加载数据集
dataset = load_dataset("lmarena-ai/VisionArena-Chat", split="train", streaming=True)
temp = []
for i, item in enumerate(dataset):
    # 处理 images 字段
    if "images" in item and item["images"]:
        for j, img in enumerate(item["images"]):
            if "bytes" in img and isinstance(img["bytes"], bytes):
                item["images"] = custom_serializer(img["bytes"])
                break

    temp.append(item)
    if i == 1000:  
        break

# 保存到 JSON 文件
with open('dataset/VisionArena_Chat/temp_data.json', 'w', encoding='utf-8') as json_file:
    json.dump(temp, json_file, ensure_ascii=False, indent=4)

print("Data saved to temp_data.json")