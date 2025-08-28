# Generation with Quantization
import logging

import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CalibConfig, QuantAlgo, QuantConfig

major, minor = torch.cuda.get_device_capability()

enable_fp8 = major > 8 or (major == 8 and minor >= 9)
enable_nvfp4 = major >= 10

quant_config_list = []

if not enable_nvfp4:
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ)
    quant_config_list.append((quant_config, None))

if enable_fp8:
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)
    calib_config = CalibConfig(
        calib_dataset="cnn_dailymail", calib_batch_size=256, calib_max_seq_length=256)

    quant_config_list.append((quant_config, calib_config))

if enable_nvfp4:
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.NVFP4, kv_cache_quant_algo=QuantAlgo.FP8)
    calib_config = CalibConfig(
        calib_dataset="cnn_dailymail", calib_batch_size=256, calib_max_seq_length=256)
    quant_config_list.append((quant_config, calib_config))
else:
    logging.error("error")


def main():
    for quant_config, calib_config in quant_config_list:
        llm = LLM(model="/data/llm_model/huggingface/TinyLlama--TinyLlama-1.1B-Chat-v1.0/", quant_config=quant_config, calib_config=calib_config
                  )
        
        # Sample prompts.
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        for prompt in prompts:
            response = llm.generate(prompt, sampling_params=sampling_params)
            print(response)
            
        llm.shutdown()
    


if __name__ == '__main__':
    main()
