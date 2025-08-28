import asyncio

from tensorrt_llm import LLM
from tensorrt_llm.sampling_params import SamplingParams


def main():
    llm = LLM(model="/data/llm_model/huggingface/TinyLlama--TinyLlama-1.1B-Chat-v1.0/")
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    
    sampling_paramers = SamplingParams(temperature=0.8, top_p=0.95)
    
    
    async def task(promt:str):
        response = llm.generate_async(promt, sampling_params=sampling_paramers)
        print(
            f"Prompt: {response.prompt!r}, Generated text: {response.outputs[0].text!r}"
        )
    
    async def main():
        tasks = [task(prompt) for prompt in prompts] 
        await asyncio.gather(*tasks)
    
    asyncio.run(main())

if __name__ == '__main__':
    main()