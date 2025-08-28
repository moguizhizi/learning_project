# stress_test_vllm.py
import time, csv, requests
import torch
from concurrent.futures import ThreadPoolExecutor

ENDPOINT = "http://127.0.0.1:8000/v1/completions"
MODEL = "qwen2_5_vl"

prompts = {
    32: "你好，请介绍一下大语言模型的基本原理。",
    512: "从Transformer、预训练语言模型、指令微调、对齐技术、推理优化等方面详细介绍一下大语言模型的发展趋势。" * 10,
    2048: "请简要解释以下技术：" + "，".join(["Transformer", "注意力机制", "混合专家", "MoE", "微调", "蒸馏", "量化"] * 100)
}

concurrency_levels = [1, 4, 16]
output_lengths = [64, 512]

def send_request(prompt, max_tokens):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    start = time.time()
    response = requests.post(ENDPOINT, headers=headers, json=payload)
    latency = time.time() - start
    return latency, response.status_code

def run_batch(prompt_len, prompt, out_len, concurrency):
    latencies = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send_request, prompt, out_len) for _ in range(concurrency)]
        for future in futures:
            latency, status = future.result()
            latencies.append(latency)
    avg_latency = sum(latencies) / len(latencies)
    return avg_latency, max(latencies), min(latencies)

def get_gpu_mem():
    try:
        import pynvml
        pynvml.nvmlInit()
        mems = []
        for i in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mems.append(mem.used / 1024 / 1024 / 1024)  # GB
        return round(max(mems), 2)
    except:
        return "N/A"

with open("vllm_perf_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["prompt_len", "output_len", "concurrency", "avg_latency", "max_latency", "min_latency", "max_gpu_mem(GB)"])
    for plen, prompt in prompts.items():
        for out_len in output_lengths:
            for cc in concurrency_levels:
                print(f"▶ 测试：prompt={plen}, out={out_len}, concurrency={cc}")
                avg, maxx, minn = run_batch(plen, prompt, out_len, cc)
                mem = get_gpu_mem()
                print(f"  - 平均延迟: {avg:.2f}s, 显存: {mem}GB")
                writer.writerow([plen, out_len, cc, round(avg,2), round(maxx,2), round(minn,2), mem])
