# 编写函数，使用openai的 api接口对多模态模型进行测试，传输一张图片和一句话
import os
import requests
import base64
import json
import time

def test_multimodal_model(image_path, text, model_url, model_name):
    # 将图片转换为base64编码
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # 构建请求数据
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "你是一个智能助理"},
                ]
            },
            # {
            #     "role": "assistant",
            #     "content": [
            #         {"type": "text", "text": "好的，我是一个智能助理"},
            #     ]
            # },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url":  f"{encoded_image}"  #f"./huochepiao.jpg"
                        # "image_url": {
                        #     "url": f"data:image/jpeg;base64,{encoded_image}"
                        # }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    # payload = {"serverURL":"http://10.20.42.137/Qwen_4_f3578828/","messages":[{"role":"user","content":[{"type":"text","text":"你好"}]},{"role":"assistant","content":"你好！有什么我可以帮助你的吗？"},{"role":"user","content":[{"type":"text","text":"1+1等于几"}]}],"deployId":"Qwen_4_f3578828"}
    
    # 发送请求
    headers = {
        "Content-Type": "application/json"
    }
    
    start_time = time.time()  # 记录开始时间
    response = requests.post(
        model_url,
        headers=headers,
        json=payload
    )
    end_time = time.time()  # 记录结束时间
    print(f"请求耗时: {end_time - start_time:.2f}秒")  # 打印耗时
    
    # 返回响应
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"API请求失败: {response.status_code}, {response.text}<<<<")
    
# 测试函数
if __name__ == "__main__":
    name = "qwen_1_0c081946"
    model_url = "http://10.20.42.147/qwen_1_0c081946/v1/chat/completions"
    pic_path = "learning_project/Tile.jpg"
    text = "请描述图片内容"
    try:
        result = test_multimodal_model(pic_path, text, model_url, name)
        print(f"result:>>>{result}")
    except Exception as e:
        print(f"发生错误: {e}")
    