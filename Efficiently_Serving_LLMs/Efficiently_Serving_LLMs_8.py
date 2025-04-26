# import asyncio
# import json
# import time
# from typing import List

# import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm
# from pydantic import BaseModel, constr
# from lorax import AsyncClient, Client
# from utils import endpoint_url, headers

# client = Client(endpoint_url, headers=headers)
# t0 = time.time()
# resp = client.generate("What is deep learning?", max_new_tokens=32)
# duration_s = time.time() - t0

# print(resp.generated_text)
# print("\n\n----------")
# print("Request duration (s):", duration_s)


from predibase import Predibase, DeploymentConfig
import os

# 初始化 Predibase 客户端
pb = Predibase(api_token="pb_xIy9iNM4Ul-cV5BTQv8k6A")

# 更新部署配置，强制启动模型
pb.deployments.update(
    deployment_ref="mistral-7b",  # 添加 deployment_ref 参数
    config=DeploymentConfig(
        base_model="mistral-7b",
        min_replicas=1,
        max_replicas=1
    )
)

# 等待模型状态变为 "Ready"
import time
for _ in range(30):  # 最多等待 5 分钟
    status = pb.deployments.get("mistral-7b").status
    print(f"Current status: {status}")
    if status == "ready":
        print("Model is ready!")
        break
    time.sleep(10)  # 每 10 秒检查一次
else:
    print("Model failed to start within 5 minutes.")
