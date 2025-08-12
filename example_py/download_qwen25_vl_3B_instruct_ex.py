# from modelscope import snapshot_download
# model_dir = snapshot_download('Qwen/Qwen3-0.6B', cache_dir='/data1/temp/llm_model/')

from modelscope.msdatasets import MsDataset 

ds =  MsDataset.load('AI-ModelScope/LLaVA-Instruct-150K', cache_dir='/data1/temp/llm_model/')

from modelscope.msdatasets import MsDataset 

ds =  MsDataset.load('lmarena-ai/VisionArena-Chat', cache_dir='/home/project/dataset')