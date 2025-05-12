# 编写函数，使用openai的 api接口对多模态模型进行测试，传输一张图片和一句话
import os
import requests
import base64
import json
import time
import cv2
import numpy as np

def to_base64(path):
    image = cv2.imdecode(np.fromfile(
        file=path, dtype=np.uint8), cv2.IMREAD_COLOR)
    # image = cv2.imread('path_to_image.jpg')
    # 将图片转换为JPG格式的字节流
    _, buffer = cv2.imencode('.jpg', image)
    # 将字节流编码为Base64字符串
    base64_image = base64.b64encode(buffer)
    # 将Base64字符串转化为Python字符串
    base64_image_str = base64_image.decode('utf-8')
    return base64_image_str

# 打开并读取txt文件
def read_txt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()  # 读取整个文件内容
            return content
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")
        return None

def test_multimodal_model_for_ascend(image_path, text, model_url, model_name):
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
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url":  f"{encoded_image}"
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    
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
        data_dict = json.loads(response.text)
        return data_dict["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API请求失败: {response.status_code}, {response.text}<<<<")
    
def test_multimodal_model_for_nvidia(image_path, text, model_url, model_name):
    # 将图片转换为base64编码
    # encoded_image = to_base64(image_path)
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
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url":f"data:image/jpeg;base64,{encoded_image}"}
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

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
        data_dict = json.loads(response.text)
        return data_dict["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API请求失败: {response.status_code}, {response.text}<<<<")
    
# 测试函数
if __name__ == "__main__":
    name = "qwen_1_c7b3def3"
    model_url = "http://10.20.42.147/qwen_1_c7b3def3/v1/chat/completions"
    pic_path = "learning_project/images/phone/Turn_on_the_Dark_theme_by_tapping_the_toggle_switch.jpg"
    text = read_txt_file("learning_project/prompt_template/phone/Turn_on_the_Dark_theme_by_tapping_the_toggle_switch.txt")
    try:
        result = test_multimodal_model_for_ascend(pic_path, text, model_url, name)
        print(f"result:>>>{result}")
        
        # 将 result 保存到 txt 文件
        output_file = "learning_project/output/Turn_on_the_Dark_theme_by_tapping_the_toggle_switch_10_20_42_147.txt"
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(str(result))  # 将 result 转换为字符串并写入
        print(f"结果已保存到 {output_file}")
    except Exception as e:
        print(f"发生错误: {e}")
        
    
    name = "/opt/models/qwen/Qwen2__5-VL-32B-Instruct"
    model_url = "http://10.20.42.106:7013/v1/chat/completions"
    pic_path = "learning_project/images/phone/Turn_on_the_Dark_theme_by_tapping_the_toggle_switch.jpg"
    text = read_txt_file("learning_project/prompt_template/phone/Turn_on_the_Dark_theme_by_tapping_the_toggle_switch.txt")
    try:
        result = test_multimodal_model_for_nvidia(pic_path, text, model_url, name)
        print(f"result:>>>{result}")
        
        # 将 result 保存到 txt 文件
        output_file = "learning_project/output/Turn_on_the_Dark_theme_by_tapping_the_toggle_switch_10_20_42_106.txt"
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(str(result))  # 将 result 转换为字符串并写入
        print(f"结果已保存到 {output_file}")
    except Exception as e:
        print(f"发生错误: {e}")
    