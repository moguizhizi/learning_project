# SPDX-License-Identifier: Apache-2.0
"""
Example online usage of Score API.

Run `vllm serve <model> --task score` to start up the server in vLLM.
"""
import argparse
import pprint

import requests
import json


def calculate_stats(score_list):
    try:
        # 检查列表是否为空
        if not score_list:
            return {
                "scores": [],
                "max": None,
                "min": None,
                "average": None,
                "var":None
            }

        # 计算统计值
        total = sum(score_list)
        count = len(score_list)
        average = total / count
        max_value = max(score_list)
        min_value = min(score_list)
        
        
        
        # 计算方差
        n = len(score_list)
        variance = sum((x - average) ** 2 for x in score_list) / (n - 1)  # 样本方差

        # 返回统计结果
        return {
            "scores": score_list,
            "max": max_value,
            "min": min_value,
            "average": average,
            "var":variance
        }

    except TypeError:
        print("错误：score_list 包含非数值元素")
        return None
    except Exception as e:
        print(f"发生未知错误：{e}")
        return None


def save_to_json(data, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"数据已保存到 {output_file}")
    except Exception as e:
        print(f"保存 JSON 文件时发生错误：{e}")


def extract_after_think(text):
    # 查找 <think> 和 </think> 的位置
    think_start = "<think>\n"
    think_end = "\n</think>"

    start_index = text.find(think_start) + len(think_start)
    end_index = text.find(think_end)

    if start_index == -1 or end_index == -1:
        return "未找到 <think> 标签或格式错误"

    # 提取 <think> 和 </think> 之间的内容
    think_content = text[start_index:end_index]

    # 提取 </think> 之后的内容
    after_think = text[end_index + len(think_end):]
    return after_think.strip()


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


def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="BAAI/bge-reranker-v2-m3")
    return parser.parse_args()


def main(args):
    api_url = f"http://{args.host}:{args.port}/score"
    model_name = "/data/llm_model/modelscope/BAAI/bge-reranker-v2-m3"

    compare_result_list = read_json_file(
        "dataset/output_file_with_whole_think.json")

    score_list = []

    for compare_result in compare_result_list:
        raw_response = compare_result["Qwen2.5_72b"]
        new_response = compare_result["Qwen3_32B"]

        answer = extract_after_think(new_response)

        text_1 = [
            raw_response
        ]
        text_2 = [
            answer
        ]
        prompt = {"model": model_name, "text_1": text_1, "text_2": text_2}
        score_response = post_http_request(prompt=prompt, api_url=api_url)
        print("\nPrompt when text_1 and text_2 are both lists:")
        pprint.pprint(prompt)
        print("\nScore Response:")
        score = score_response.json()["data"][0]["score"]
        pprint.pprint(score)
        score_list.append(score)

    save_to_json(calculate_stats(score_list),
                 "dataset/output_file_with_whole_think_score.json")
    
    
    compare_result_list = read_json_file(
        "dataset/output_file_without_whole_think.json")

    score_list = []

    for compare_result in compare_result_list:
        raw_response = compare_result["Qwen2.5_72b"]
        new_response = compare_result["Qwen3_32B"]

        answer = new_response.replace("<think>", "").strip()

        

        text_1 = [
            raw_response
        ]
        text_2 = [
            answer
        ]
        prompt = {"model": model_name, "text_1": text_1, "text_2": text_2}
        score_response = post_http_request(prompt=prompt, api_url=api_url)
        print("\nPrompt when text_1 and text_2 are both lists:")
        pprint.pprint(prompt)
        print("\nScore Response:")
        score = score_response.json()["data"][0]["score"]
        
        pprint.pprint(score)
        score_list.append(score)

    save_to_json(calculate_stats(score_list),
                 "dataset/output_file_without_whole_think_score.json")


if __name__ == "__main__":
    args = parse_args()
    main(args)
