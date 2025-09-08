from vllm import LLM, SamplingParams
from modelscope import AutoProcessor
from qwen_vl_utils import process_vision_info

# 1. 加载模型
llm = LLM(
    model="/home/llm_model/Qwen/Qwen2.5-VL-3B-Instruct",
    tensor_parallel_size=1,
    trust_remote_code=True,
    dtype="bfloat16",
)

# 2. Processor
processor = AutoProcessor.from_pretrained("/home/llm_model/Qwen/Qwen2.5-VL-3B-Instruct")

# 3. 输入消息
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# 4. prompt
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 5. 处理图像/视频
image_inputs, video_inputs = process_vision_info(messages)

multi_modal_data = {}
if image_inputs is not None:
    multi_modal_data["image"] = image_inputs
if video_inputs is not None:
    multi_modal_data["video"] = video_inputs

# 6. 采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=128,
)

# 7. 调用 vLLM
outputs = llm.generate(
    {"prompt": prompt, "multi_modal_data": multi_modal_data},
    sampling_params
)

# 8. 打印结果
for output in outputs:
    print(output.outputs[0].text)
