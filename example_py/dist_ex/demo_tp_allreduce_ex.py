# demo_tp_allreduce_ex.py
import os
import torch
import torch.distributed as dist

from vllm.distributed import init_distributed_environment
from vllm.distributed import parallel_state as ps
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce


def main():
    # 读取环境变量（torchrun 会注入）
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    torch.cuda.set_device(local_rank)

    # 1) 初始化全局进程组
    init_distributed_environment(backend="nccl")

    # 2) 初始化 vLLM 的并行拓扑（建立 TP 组）
    #    这里假设我们只做 TP，不做 PP（pipeline），所以 pp_size=1
    tp_size = 2                       # 跟 --nproc_per_node 一致
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
    )

    # （可选）查看当前进程在 TP 组内的信息
    tp_rank = ps.get_tensor_model_parallel_rank()
    tp_world_size = ps.get_tensor_model_parallel_world_size()

    # 3) 各 rank 上放一个不同的张量
    x = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
    print(f"[Before] global_rank={rank}, local_rank={local_rank}, "
          f"tp_rank={tp_rank}/{tp_world_size}: x={x.item()}")

    # 4) 只在 TP 组内做 all-reduce（vLLM 封装）
    y = tensor_model_parallel_all_reduce(x)

    print(f"[After ] global_rank={rank}, local_rank={local_rank}, "
          f"tp_rank={tp_rank}/{tp_world_size}: y={y.item()}")

    # 5) 关闭并行状态与进程组
    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
