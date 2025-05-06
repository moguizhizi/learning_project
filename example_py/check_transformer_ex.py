from vllm import LLM
llm = LLM(model="facebook/opt-125m", task="generate")  # Name or path of your model
llm.apply_model(lambda model: print(type(model)))