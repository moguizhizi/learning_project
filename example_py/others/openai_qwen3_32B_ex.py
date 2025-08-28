# 编写函数，使用openai的 api接口对多模态模型进行测试，传输一张图片和一句话
import os
import requests
import base64
import json
import time
import cv2
import numpy as np


def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # 读取 JSON 文件并解析为字典
            return data
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在。")
        return None
    except json.JSONDecodeError:
        print(f"错误：文件 '{file_path}' 不是有效的 JSON 格式。")
        return None
    except Exception as e:
        print(f"发生未知错误：{e}")
        return None


def test_multimodal_model_for_ascend(system_text, prompt_planning, model_url, model_name):

    # 构建请求数据
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system",
             "content": system_text},
            {
                "role": "user",
                "content": prompt_planning
            }
        ],
        "temperature": 0.0,
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
        raise Exception(
            f"API请求失败: {response.status_code}, {response.text}<<<<")


# 测试函数
if __name__ == "__main__":
    name = "Qwen3-32B"
    model_url = "http://10.20.42.105/Qwen_local_57_352a2d07/v1/chat/completions"
    output_steps_list = read_json_file("dataset/output_steps.json")
    
    compare_list = []

    try:
        for step_dict in output_steps_list:
            prompt_planning = step_dict["prompt_planning"]
            raw_response = step_dict["raw_response"]
            system_text = step_dict["system_text"]

            result = test_multimodal_model_for_ascend(
                system_text, prompt_planning, model_url, name)
            print(f"result:>>>{result}")
            
            temp_dict = {"Qwen2.5_72b": raw_response, "Qwen3_32B": result}
            compare_list.append(temp_dict)

        # 保存 compare_list 到 JSON 文件
        output_file = "dataset/compare_results.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(compare_list, file, ensure_ascii=False, indent=4)
            print(f"对比结果已保存到 {output_file}")
        except Exception as e:
            print(f"保存 JSON 文件时发生错误：{e}")

    except Exception as e:
        print(f"发生错误: {e}")