from qwen_vl_utils import process_vision_info

# 构造消息
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image", "image": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"}
        ]
    }
]

# 处理视觉信息
updated_messages, vision_inputs = process_vision_info(messages)

print("Updated Messages:", updated_messages)
print("Vision Inputs:", vision_inputs)