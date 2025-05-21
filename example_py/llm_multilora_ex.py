from huggingface_hub import snapshot_download

from tensorrt_llm import LLM, BuildConfig
from tensorrt_llm.executor import LoRARequest
from tensorrt_llm.lora_manager import LoraConfig


def main():
    lora_dir1 = snapshot_download(repo_id="snshrivas10/sft-tiny-chatbot",
                                  local_dir="/data/llm_lora/huggingface/snshrivas10/sft-tiny-chatbot/")
    lora_dir2 = snapshot_download(
        repo_id="givyboy/TinyLlama-1.1B-Chat-v1.0-mental-health-conversational", local_dir="/data/llm_lora/huggingface/givyboy/TinyLlama-1.1B-Chat-v1.0-mental-health-conversational/")
    lora_dir3 = snapshot_download(repo_id="barissglc/tinyllama-tarot-v1",
                                  local_dir="/data/llm_lora/huggingface/barissglc/tinyllama-tarot-v1")

    build_config = BuildConfig()
    build_config.lora_config = LoraConfig(
        lora_dir=[lora_dir1], max_lora_rank=64)

    llm = LLM(model="/data/llm_model/huggingface/TinyLlama--TinyLlama-1.1B-Chat-v1.0/",
              enable_lora=True, build_config=build_config)

    LoRARequest

    # Sample prompts
    prompts = [
        "Hello, tell me a story: ",
        "Hello, tell me a story: ",
        "I've noticed you seem a bit down lately. Is there anything you'd like to talk about?",
        "I've noticed you seem a bit down lately. Is there anything you'd like to talk about?",
        "In this reading, the Justice card represents a situation where",
        "In this reading, the Justice card represents a situation where",
    ]

    outputs = llm.generate(prompts, lora_request=[None, LoRARequest("A", lora_int_id=1, lora_path=lora_dir1), None, LoRARequest(
        "B", lora_int_id=2, lora_path=lora_dir2), None, LoRARequest("C", lora_int_id=3, lora_path=lora_dir3)])
    
    for output in outputs:
        print(output.outputs[0].text)


if __name__ == '__main__':
    main()
