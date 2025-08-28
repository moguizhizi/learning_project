from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch


model = AutoModelForCausalLM.from_pretrained(
        "/home/temp/llm_model/Qwen/Qwen3-8B", device_map="auto")

for name, module in model.named_modules():
    print(f"name:{name}")
    print(f"module:{module}")

# from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info

# # default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "/home/temp/llm_model/nm-testing/Qwen2___5-VL-72B-Instruct-quantized___w8a8", torch_dtype="auto", device_map="auto"
# )

# for name, module in model.named_modules():
#     print(f"name:{name}")
#     print(f"module:{module}")
