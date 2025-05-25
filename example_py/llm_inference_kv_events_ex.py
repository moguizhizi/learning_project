from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.llmapi import KvCacheConfig


def main():
    pytorch_config = PyTorchConfig(
        autotuner_enabled=True, kv_cache_dtype="auto")
    llm = LLM(model="/data/llm_model/huggingface/TinyLlama--TinyLlama-1.1B-Chat-v1.0/", 
              tensor_parallel_size=2,
              pytorch_backend_config=pytorch_config,
              kv_cache_config=KvCacheConfig(enable_block_reuse=True, event_buffer_max_size=1024),
              backend="pytorch")
    
    common_prefix = (
        "After the ghost's departure, Barnardo notes Horatio's pale appearance and asks if he's okay. "
        "Horatio concedes that he's shaken and confesses that, without witnessing the ghost himself, he wouldn't have believed it existed. "
        "He's also disturbed by the ghost's striking resemblance to the king. It even seems to be wearing the former king's armor. "
        "Horatio thinks the ghost's presence foretells that something is about to go wrong in Denmark. "
        "Marcellus concurs with Horatio, as he and the other guards have observed that their schedules have become more rigorous and have also noticed the preparations taking place within Elsinore, including the building of cannons, the storing of weapons, and the preparation of ships."
    )
    prompts = [
        common_prefix, common_prefix + " Marcellus also notes that the king's"
    ]
    
    outputs = llm.generate(prompts, sampling_params=SamplingParams(temperature=0.0, top_p=0.85))
    for output in outputs:
        print(output.outputs[0].text)
    
    result = llm.get_kv_cache_events(10)
    print(result)


if __name__ == '__main__':
    main()
