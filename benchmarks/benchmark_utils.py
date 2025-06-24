# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import math
import os
from typing import Any
import re
import ast
from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def convert_to_pytorch_benchmark_format(
    args: argparse.Namespace, metrics: dict[str, list], extra_info: dict[str, Any]
) -> list:
    """
    Save the benchmark results in the format used by PyTorch OSS benchmark with
    on metric per record
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    records = []
    if not os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        return records

    for name, benchmark_values in metrics.items():
        record = {
            "benchmark": {
                "name": "vLLM benchmark",
                "extra_info": {
                    "args": vars(args),
                },
            },
            "model": {
                "name": args.model,
            },
            "metric": {
                "name": name,
                "benchmark_values": benchmark_values,
                "extra_info": extra_info,
            },
        }

        tp = record["benchmark"]["extra_info"]["args"].get("tensor_parallel_size")
        # Save tensor_parallel_size parameter if it's part of the metadata
        if not tp and "tensor_parallel_size" in extra_info:
            record["benchmark"]["extra_info"]["args"]["tensor_parallel_size"] = (
                extra_info["tensor_parallel_size"]
            )

        records.append(record)

    return records


class InfEncoder(json.JSONEncoder):
    def clear_inf(self, o: Any):
        if isinstance(o, dict):
            return {k: self.clear_inf(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self.clear_inf(v) for v in o]
        elif isinstance(o, float) and math.isinf(o):
            return "inf"
        return o

    def iterencode(self, o: Any, *args, **kwargs) -> Any:
        return super().iterencode(self.clear_inf(o), *args, **kwargs)


def write_to_json(filename: str, records: list) -> None:
    with open(filename, "w") as f:
        json.dump(
            records,
            f,
            cls=InfEncoder,
            default=lambda o: f"<{type(o).__name__} object is not JSON serializable>",
        )

def extract_json_block(text: str) -> str:
    """
    从包含 Markdown 格式 ```json 代码块的文本中提取 JSON 字符串。

    参数:
        text (str): 包含 JSON 代码块的字符串文本。

    返回:
        str: 提取到的 JSON 内容字符串（原始格式，未解析）。

    异常:
        ValueError: 如果未找到 JSON 代码块。
    """
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if not match:
        raise ValueError("未找到 json 代码块")

    return match.group(1).strip()


def cosine_similarity_between_texts(text1: str, text2: str) -> float:
    # 向量化两个文本
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # 计算余弦相似度
    sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return sim_matrix[0][0]  # 返回一个浮点数：0 ~ 1


def calculate_accuracy(prediction, ground_true, dataset_name):
    if  ground_true is None:
        return 0.0
    
    if dataset_name == "phonetest": 
       json_block = extract_json_block(prediction)
       
       return cosine_similarity_between_texts(json_block, ground_true)
    else:
        return 0.0