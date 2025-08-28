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
    rank = dist.get_rank()
    print(f"进程 {rank}: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"进程 {rank}: GPU 数量={torch.cuda.device_count()}")
    print(f"进程 {rank}: 当前设备={torch.cuda.current_device()}")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size
    )
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(f"进程 {rank} 分配到设备 {device}")
    get_cuda_rng_tracker().add("model-parallel-rng", seed)

# 自定义 MLP 类以打印中间信息
class DebugMLP(MLP):
    def forward(self, x):
        rank = dist.get_rank()
        print(f"进程 {rank}: MLP 输入形状: {x.shape}")
        
        # 调用第一个线性层 (ColumnParallelLinear)，解包输出
        hidden, fc1_bias = self.linear_fc1(x)  # 解包输出和偏置
        print(f"进程 {rank}: ColumnParallelLinear (fc1) 输出形状: {hidden.shape}")
        print(f"进程 {rank}: ColumnParallelLinear (fc1) 偏置: {fc1_bias if fc1_bias is not None else 'None'}")
        
        # 激活函数
        hidden = self.activation_func(hidden)
        print(f"进程 {rank}: 激活函数后形状: {hidden.shape}")
        
        # 调用第二个线性层 (RowParallelLinear)，解包输出
        output, fc2_bias = self.linear_fc2(hidden)  # 解包输出和偏置
        print(f"进程 {rank}: RowParallelLinear (fc2) 输出形状: {output.shape}")
        print(f"进程 {rank}: RowParallelLinear (fc2) 偏置: {fc2_bias if fc2_bias is not None else 'None'}")
        
        # 返回输出（不返回 self.bias，因为它可能不存在）
        return output, fc2_bias  # 返回 fc2 的偏置（如果存在）

# 主函数
if __name__ == "__main__":
    # 设置环境变量以优化性能
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    try:
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

        # 定义 mlp_spec，使用自定义 DebugMLP
        mlp_spec = ModuleSpec(
            module=DebugMLP,
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
            )
        )

        # 构建 MLP 模块
        mlp = build_module(mlp_spec, config=config)

        # 访问 ColumnParallelLinear 和 RowParallelLinear 层
        rank = dist.get_rank()
        linear_fc1 = mlp.linear_fc1
        linear_fc2 = mlp.linear_fc2

        # 打印层的权重和偏置信息
        print(f"进程 {rank}: ColumnParallelLinear (fc1) 权重形状: {linear_fc1.weight.shape}")
        if linear_fc1.bias is not None:
            print(f"进程 {rank}: ColumnParallelLinear (fc1) 偏置形状: {linear_fc1.bias.shape}")
        print(f"进程 {rank}: RowParallelLinear (fc2) 权重形状: {linear_fc2.weight.shape}")
        if linear_fc2.bias is not None:
            print(f"进程 {rank}: RowParallelLinear (fc2) 偏置形状: {linear_fc2.bias.shape}")

        # 创建输入张量
        input_tensor = torch.randn(4, 10, config.hidden_size).cuda()
        print(f"进程 {rank}: 输入张量形状: {input_tensor.shape}")

        # 前向传播
        output, output_bias = mlp(input_tensor)
        print(f"进程 {rank}: 输出张量形状: {output.shape}")
        print(f"进程 {rank}: 输出偏置: {output_bias if output_bias is not None else 'None'}")

    except Exception as e:
        print(f"进程 {dist.get_rank()}: 发生异常: {e}")
        raise
    finally:
        # 确保分布式进程组被清理
        if dist.is_initialized():
            dist.destroy_process_group()