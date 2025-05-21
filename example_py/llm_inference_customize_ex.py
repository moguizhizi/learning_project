from tensorrt_llm.llmapi import LLM, KvCacheConfig, SamplingParams, BuildConfig


def main():
    build_config = BuildConfig()
    build_config.max_batch_size = 128
    build_config.max_num_tokens = 1000
    build_config.max_beam_width = 4

    llm = LLM(model="/data/llm_model/huggingface/TinyLlama--TinyLlama-1.1B-Chat-v1.0/",
              build_config=build_config, kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.8))
    
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.85, n = 4, use_beam_search=True)
    for prompt in prompts:
        response = llm.generate(prompt, sampling_params=sampling_params)
        print(response.outputs[0])


if __name__ == '__main__':
    main()
