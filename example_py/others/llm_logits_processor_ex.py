import torch
from typing import List, Optional

from tensorrt_llm import LLM
from tensorrt_llm.sampling_params import BatchedLogitsProcessor, LogitsProcessor, SamplingParams


class MyLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_id):
        super().__init__()
        self.allowed_token_id = allowed_token_id

    def __call__(self, req_ids: List[int], logits: List[torch.Tensor],
                 token_ids: List[List[List[int]]], stream_ptr: int,
                 client_ids: List[Optional[int]]) -> None:
        mask = torch.full_like(logits, device="cpu", fill_value=float("-inf"))
        mask[:, :, self.allowed_token_id] = 0
        
        stream = torch.cuda.ExternalStream(stream_ptr)
        with torch.cuda.stream(stream):
            mask = mask.to(logits.device, non_blocking=True)
            logits += mask


class MyBatchedLogitsProcessor(BatchedLogitsProcessor):
    def __init__(self, allowed_token_id):
        super().__init__()
        self.allowed_token_id = allowed_token_id

    def __call__(self, req_ids: List[int], logits: List[torch.Tensor],
                 token_ids: List[List[List[int]]], stream_ptr: int,
                 client_ids: List[Optional[int]]) -> None:
        masks = []
        for req_id, req_logits, req_token_ids, client_id in zip(
                req_ids, logits, token_ids, client_ids):
            mask = torch.full_like(req_logits,
                                   fill_value=float("-inf"),
                                   device="cpu")
            mask[:, :, self.allowed_token_id] = 0
            masks.append(mask)

        # Move masks to device and add to logits using non-blocking operations
        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            for req_logits, mask in zip(logits, masks):
                req_logits += mask.to(req_logits.device, non_blocking=True)


def main():
    llm = LLM(model="/data/llm_model/huggingface/TinyLlama--TinyLlama-1.1B-Chat-v1.0/",
              batched_logits_processor=MyBatchedLogitsProcessor(allowed_token_id=42))
    # Sample prompts
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
    ]

    for prompt_id, prompt in enumerate(prompts):
        if prompt_id % 2 == 0:
            sampling_paramas = SamplingParams(temperature=0.8, top_p=0.9)
            response = llm.generate(prompt, sampling_params=sampling_paramas)
            print(f"unused logits processor:{response}")
        else:
            sampling_paramas = SamplingParams(
                temperature=0.8, top_p=0.9, logits_processor=MyLogitsProcessor(allowed_token_id=42))
            response = llm.generate(prompt, sampling_params=sampling_paramas)
            print(f"used logits processor:{response}")

    sampling_params = SamplingParams(apply_batched_logits_processor=True)
    for output in llm.generate(prompts, sampling_params):
        print(
            f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
        )


if __name__ == '__main__':
    main()
