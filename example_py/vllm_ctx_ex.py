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

ctx = llm.llm_engine.model_executor.driver_worker.worker.compilation_config.static_forward_context

# layer_need_kv_cache = []
# for layer_name in ctx:
    # if ctx[layer_name].attn_type in (AttentionType.DECODER, AttentionType.ENCODER_DECODER):
    #         layer_need_kv_cache.append(layer_name)
    # print(layer_name)

for layer_name in ctx:
    layer_config = ctx[layer_name]
    print(f"Layer: {layer_name}")
    print(f"  Attention Type: {layer_config.attn_type}")  # 可能是DECODER/ENCODER等

model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()
vllm_names = list(dict(model.named_parameters()).keys())
print('\n'.join(vllm_names))

for name, param in sorted(model.named_parameters()):
    param