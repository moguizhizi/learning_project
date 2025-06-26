import os
from vllm import LLM, SamplingParams
from vllm.inputs import (ExplicitEncoderDecoderPrompt, TextPrompt,
                         TokensPrompt, zip_enc_dec_prompts)

# 设置环境变量 VLLM_USE_V1=1
os.environ["VLLM_USE_V1"] = "0"

# Create a BART encoder/decoder model instance
llm = LLM(
    # model="facebook/bart-large-cnn",
    model="/home/temp/llm_model/Qwen/Qwen2.5-VL-7B-Instruct",
    max_model_len=64,
)

llm.llm_engine.model_executor.driver_worker.worker.compilation_config.static_forward_context