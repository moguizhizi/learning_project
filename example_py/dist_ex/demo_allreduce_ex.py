import os
import torch
import torch.distributed as dist


# torchrun --nproc_per_node=2 demo_allreduce_ex.py

def main():
    # 初始化进程组（NCCL 后端）
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()          # 当前进程号 0 或 1
    torch.cuda.set_device(rank)     # 绑定到 GPU rank

    # 每个进程各有一张不同的张量
    x = torch.tensor([rank + 1.0], device='cuda')  # GPU0: [1]  GPU1: [2]

    print(f"before rank{rank}: {x.item()}")

    # all-reduce：组内求和，结果写回 x
    dist.all_reduce(x, op=dist.ReduceOp.SUM)   # 等价于 dist.all_reduce(x)

    print(f"after  rank{rank}: {x.item()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
