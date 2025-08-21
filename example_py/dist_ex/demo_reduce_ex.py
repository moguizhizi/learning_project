import torch
import torch.distributed as dist


# 启动方式：
#   torchrun --nproc_per_node=2 demo_reduce_ex.py

def main():
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    # 每个 rank 都有自己的张量
    x = torch.tensor([rank + 1.0], device="cuda")   # rank0: [1], rank1: [2]

    print(f"[before] rank{rank}: {x.item()}")

    # reduce：把所有 rank 的张量 sum，并写回 dst=0 的张量
    dist.reduce(x, dst=0, op=dist.ReduceOp.SUM)

    # 打印结果
    print(f"[after ] rank{rank}: {x.item()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
