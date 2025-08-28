import torch.distributed as dist

from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

import os
import torch


class DebugMLP(MLP):
    def __init__(self, config, submodules, is_expert=False, input_size=None):
        super().__init__(config, submodules, is_expert, input_size)

    def forward(self, x):
        rank = dist.get_rank()
        print(f"进程{rank} 的输入张量大小：{x.size()}")
        
        hidden, bias = self.linear_fc1(x)
        print(f"进程{rank} ColumnParallelLinear 后张量大小：{hidden.size()}")
        print(f"进程{rank} ColumnParallelLinear 后bias大小：{bias.size() if bias is not None else 0}")
        
        hidden = self.activation_func(hidden)
        print(f"进程{rank}  activation func 后张量大小：{hidden.size()}")
        
        hidden, bias = self.linear_fc2(hidden)
        print(f"进程{rank} RowParallelLinear 后张量大小：{hidden.size()}")
        print(f"进程{rank} RowParallelLinear 后bias大小：{bias.size() if bias is not None else 0}")
        
        return hidden, bias


def init_distribution_state(tensor_model_parallel_size, seed):
    dist.init_process_group(backend="nccl")

    count = torch.cuda.device_count()

    print(f'CUDA_VISIBLE_DEVICES:{os.environ.get("CUDA_VISIBLE_DEVICES")}')
    print(f'当前GPU的数量:{count}')
    print(f'当前device:{torch.cuda.current_device()}')

    rank = dist.get_rank()
    device = rank % count
    torch.cuda.set_device(device)
    print(f"进程{rank} 被移到了device:{device}")

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size)

    get_cuda_rng_tracker().add("model-parallel-rng", seed)


if __name__ == '__main__':
    # 设置环境变量以优化性能
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    try:
        tensor_model_parallel_size = 2
        init_distribution_state(tensor_model_parallel_size, 1234)

        config = TransformerConfig(
            num_layers=1,
            num_attention_heads=8,
            hidden_size=512,
            ffn_hidden_size=2048,
            hidden_dropout=0.8,
            tensor_model_parallel_size=tensor_model_parallel_size)
        mlp_spec = ModuleSpec(
            module=DebugMLP,
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear
            ))

        mlp_instance = build_module(mlp_spec, config=config)
        linear_fc1 = mlp_instance.linear_fc1
        linear_fc2 = mlp_instance.linear_fc2

        rank = dist.get_rank()
        print(
            f"进程 {rank} ColumnParallelLinear Weight Size:{linear_fc1.weight.shape}")
        print(f"进程 {rank} RowParallelLinear Weight Size:{linear_fc2.weight.shape}")
        
        input_tensor = torch.randn(4,10,config.hidden_size).cuda()
        print(f"输入的Tensor 大小为:{input_tensor.size()}")
        
        output, bias = mlp_instance(input_tensor)
        print(f"进程 {rank} mlp size:{output.shape}")
        
        
    except Exception as e:
        print(f"进程 {dist.get_rank()}: 发生异常: {e}")
        raise
    finally:
        # 确保分布式进程组被清理
        if dist.is_initialized():
            dist.destroy_process_group()
