from typing import Literal

QueryModality = Literal["text", "image", "text+image"]


def process_query(modality: QueryModality) -> None:
    print(f"Processing {modality} query")

process_query("text")         # 合法
process_query("text+image")   # 合法
process_query("audio")        # 非法，mypy 报错