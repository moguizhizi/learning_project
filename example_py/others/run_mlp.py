import torch
import torch.distributed as dist
import os
from megatron.core import parallel_state
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

# 初始化分布式环境
def initialize_megatron_model_parallel(tensor_model_parallel_size=2, seed=1234):
    dist.init_process_group(backend='nccl')
    # 打印可用 GPU 数量和当前设备
    rank = dist.get_rank()
    print(f"Rank {rank}: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Rank {rank}: Number of GPUs={torch.cuda.device_count()}")
    print(f"Rank {rank}: Current device={torch.cuda.current_device()}")
    # 初始化 Megatron 的张量并行
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size
    )
    # 手动设置 GPU
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(f"Rank {rank} assigned to device {device}")
    # 添加 RNG 状态
    get_cuda_rng_tracker().add("model-parallel-rng", seed)

# 主函数
if __name__ == "__main__":
    # 设置张量并行大小
    tensor_parallel_size = 2
    initialize_megatron_model_parallel(tensor_parallel_size, seed=1234)

    # 定义 Transformer 配置
    config = TransformerConfig(
        num_layers=1,
        num_attention_heads=8,
        hidden_size=512,
        hidden_dropout=0.1,
        ffn_hidden_size=2048,
        tensor_model_parallel_size=tensor_parallel_size
    )

    # 定义一个简单的 mlp_spec
    mlp_spec = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=ColumnParallelLinear,
            linear_fc2=RowParallelLinear,
        )
    )

    # 构建 MLP 模块
    mlp = build_module(mlp_spec, config=config)

    # 创建输入张量
    input_tensor = torch.randn(4, 10, config.hidden_size).cuda()
    print(f"Input tensor shape: {input_tensor.shape}")

    # 前向传播，解包输出
    output, output_bias = mlp(input_tensor)
    print(f"Output tensor shape: {output.shape}")
    print(f"Output bias: {output_bias}")

    # 清理分布式环境
    dist.destroy_process_group()