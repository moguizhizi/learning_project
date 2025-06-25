from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import torch.nn as nn

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/home/temp/llm_model/Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# Store the model parameters on CPU
cpu_model = {}
for name, params in model.named_parameters():
    cpu_model[name] = torch.empty_like(params, device="cpu")

def offload_model_weights():
    for name, params in model.named_parameters():
        # Offload the weights to CPU
        params.data = cpu_model[name]
        
        # Debugging: Print the device of each parameter to verify the offloading
        print(f"Parameter {name} is now on device: {params.data.device}")

# Offload the weights
offload_model_weights()

# Check if weights are correctly moved to CPU by printing some examples
for name, params in model.named_parameters():
    print(f"After offloading, {name} is on device: {params.data.device}")
