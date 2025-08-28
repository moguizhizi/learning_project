# from vllm import LLMEngine, EngineArgs
# from vllm.sampling_params import SamplingParams
# from vllm.utils import random_uuid

# # 初始化 engine
# engine_args = EngineArgs(model="mistralai/Mistral-7B-Instruct-v0.2", max_model_len=2048)
# engine = LLMEngine.from_engine_args(engine_args)

# # 创建请求
# prompt = "Hello, how are you?"
# request_id = random_uuid()
# sampling_params = SamplingParams(temperature=0.7, top_p=0.9)

# # 提交请求
# engine.add_request(
#     request_id=request_id,
#     prompt=prompt,
#     params=sampling_params
# )

# # 轮询获取输出
# while True:
#     outputs = engine.step()
#     if outputs:
#         for output in outputs:
#             print(">>>", output.outputs[0].text.strip())
#         break
    
# engine.add_request(
#     request_id="user_001_request_001",
#     prompt="Hi, what’s the weather today?",
#     params=sampling_params
# )

# engine.add_request(
#     request_id="user_002_request_001",
#     prompt="Tell me a joke!",
#     params=sampling_params
# )

# pending_requests = {
#     "user_001_request_001",
#     "user_002_request_001"
# }

# while pending_requests:
#     outputs = engine.step()
#     for output in outputs:
#         req_id = output.request_id
#         if req_id in pending_requests:
#             print(f"Request {req_id} -> {output.outputs[0].text.strip()}")
#             pending_requests.remove(req_id)


from vllm import EngineArgs, LLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

engine_args = EngineArgs(model="/data1/temp/huggingface/Mistral-7B-Instruct-v0.2", max_model_len=2048)
llm_engine = LLMEngine.from_engine_args(engine_args)

sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=256)

prompt = "What is it?"
request_id = random_uuid()

llm_engine.add_request(prompt=prompt, request_id=request_id, params=sampling_params)

while True:
    outputs = llm_engine.step()
    if outputs:
        for output in outputs:
            print(f">> {output.outputs[0].text.strip()}")
        break


prompt = "What is it?"
request_id_1 = random_uuid()

llm_engine.add_request(prompt=prompt, request_id=request_id_1, params=sampling_params)


prompt = "What is it?"
request_id_2 = random_uuid()

llm_engine.add_request(prompt=prompt, request_id=request_id_2, params=sampling_params)


temp_list = [request_id_1, request_id_2]

while temp_list:
    outputs = llm_engine.step()
    if outputs:
        for output in outputs:
            print(f">> {output.outputs[0].text.strip()}")
        temp_list.pop(0)

