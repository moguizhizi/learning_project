

import os
from vllm import LLM, SamplingParams

os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3"

prompts = [
    "Hello, my name is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(
        model="/home/llm_model/neuralmagic/Qwen2.5-VL-3B-Instruct-quantized.w8a8",
        max_model_len=26240,
        load_format="runai_streamer",
        pipeline_parallel_size=1,
        tensor_parallel_size=4,      # 纯 PP 时设为 1
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")