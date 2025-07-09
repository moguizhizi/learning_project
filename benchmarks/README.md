<details>
<summary><b>📈 Example - Offline Benchmark</b></summary>

<br/>

First start serving your model

```bash
vllm serve /home/temp/llm_model/nm-testing/Qwen2.5-VL-72B-Instruct-quantized.w8a8 --load-format runai_streamer --tensor-parallel-size 4 --max-model-len 8192 --max-num-seqs 2048 --kv-cache auto --gpu-memory-utilization 0.5 --disable-custom-all-reduce --served-model-name qwen2_5_vl_72B_quant --disable-log-requests
```

```bash
python3 learning_project/benchmarks/benchmark_serving.py --backend openai-chat --model /home/temp/llm_model/nm-testing/Qwen2___5-VL-72B-Instruct-quantized___w8a8 --served-model-name qwen2_5_vl_72B_quant --endpoint /v1/chat/completions --dataset-name phonetest --dataset-path /home/project/dataset/phonetest/web_nj_action_0426_grpo.json --num-prompts 10 --result_dir /home/project/learning_project/benchmarks/output/qwen2_5_vl_72B --save-result
```

If successful, you will see the following output

```
============ Serving Benchmark Result ============
Successful requests:                     10
Benchmark duration (s):                  90.66
Total input tokens:                      44012
Total generated tokens:                  1462
Request throughput (req/s):              0.11
Output token throughput (tok/s):         16.13
Total Token throughput (tok/s):          501.56
---------------Time to First Token----------------
Mean TTFT (ms):                          43073.78
Median TTFT (ms):                        43307.82
P99 TTFT (ms):                           83289.33
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          49.13
Median TPOT (ms):                        50.31
P99 TPOT (ms):                           51.96
---------------Inter-token Latency----------------
Mean ITL (ms):                           48.60
Median ITL (ms):                         43.66
P99 ITL (ms):                            55.69
---------------------Accuracy---------------------
Accuracy Rate (%):                       80.00
==================================================
```


```bash
python3 learning_project/benchmarks/benchmark_serving.py --backend openai-chat --model /home/temp/llm_model/Qwen/Qwen2.5-VL-72B-Instruct --served-model-name /opt/models/Qwen2.5-VL-72B-Instruct --endpoint /v1/chat/completions --dataset-name phonetest --dataset-path /home/project/dataset/phonetest/web_nj_action_0426_grpo.json --num-prompts 32 --result_dir /home/project/learning_project/benchmarks/output/qwen2_5_vl_72B --save-result --same --host 10.20.42.106 --port 7013
```

If successful, you will see the following output

```
============ Serving Benchmark Result ============
Successful requests:                     16
Benchmark duration (s):                  86.44
Total input tokens:                      75216
Total generated tokens:                  2511
Request throughput (req/s):              0.19
Output token throughput (tok/s):         29.05
Total Token throughput (tok/s):          899.17
---------------Time to First Token----------------
Mean TTFT (ms):                          43533.17
Median TTFT (ms):                        39528.12
P99 TTFT (ms):                           71475.48
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          173.70
Median TPOT (ms):                        176.11
P99 TPOT (ms):                           321.31
---------------Inter-token Latency----------------
Mean ITL (ms):                           168.06
Median ITL (ms):                         81.52
P99 ITL (ms):                            2160.13
---------------------Accuracy---------------------
Accuracy Rate (%):                       100.00
==================================================
```
</details>