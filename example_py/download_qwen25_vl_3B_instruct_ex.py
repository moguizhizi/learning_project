from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', cache_dir='/data/llm_model/modelscope')