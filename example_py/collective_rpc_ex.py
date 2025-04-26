# collective_rpc_demo.py
import ray
import numpy as np
from typing import Dict

# 1. 定义一个 Ray Actor 用于管理集体通信
@ray.remote(num_gpus=1)  # 每个 Actor 占用 1 个 GPU
class CollectiveManager:
    def __init__(self):
        self.weights = None
        self.group = {}  # 模拟通信组

    # 初始化通信组
    def init_weight_update_group(self, master_addr: str, master_port: int, rank: int, world_size: int) -> str:
        self.group = {
            "master": f"{master_addr}:{master_port}",
            "rank": rank,
            "world_size": world_size
        }
        return f"Rank {rank} initialized with master at {master_addr}:{master_port}"

    # 全局权重同步（模拟 all_reduce）
    def sync_weights(self, local_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.group:
            raise RuntimeError("Communication group not initialized!")
        
        # 模拟 AllReduce 操作（实际场景用 NCCL/GRPC）
        print(f"Rank {self.group['rank']} syncing weights...")
        averaged_weights = {k: v / self.group["world_size"] for k, v in local_weights.items()}
        return averaged_weights

# 2. 主程序
def main():
    # 初始化 Ray
    ray.init(num_gpus=2, num_cpus=4)

    # 创建两个 CollectiveManager Actor (模拟两个 GPU 进程)
    manager1 = CollectiveManager.remote()
    manager2 = CollectiveManager.remote()

    # 初始化通信组
    master_addr = "127.0.0.1"
    master_port = 12345
    
    # 异步调用初始化
    init1 = manager1.init_weight_update_group.remote(master_addr, master_port, rank=0, world_size=2)
    init2 = manager2.init_weight_update_group.remote(master_addr, master_port, rank=1, world_size=2)
    
    # 等待初始化完成
    print(ray.get(init1))
    print(ray.get(init2))

    # 模拟本地权重（两个节点各有不同初始值）
    weights1 = {"layer1": np.array([1.0, 2.0], dtype=np.float32)}
    weights2 = {"layer1": np.array([3.0, 4.0], dtype=np.float32)}

    # 同步权重（集体通信）
    sync1 = manager1.sync_weights.remote(weights1)
    sync2 = manager2.sync_weights.remote(weights2)

    # 打印结果（应为平均值）
    print("Synced weights (Rank 0):", ray.get(sync1))
    print("Synced weights (Rank 1):", ray.get(sync2))

    ray.shutdown()

if __name__ == "__main__":
    main()