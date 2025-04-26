import ray
import torch.distributed as dist
import torch


@ray.remote(num_gpus=1)
class LLMWork():
    def __init__(self, work_id):
        self.work_id = work_id

    def get_work_id(self):
        return self.work_id

    def collective_rpc(self, method: str, master_ip: str, master_port: int, num_works: int, rank: int):
        self.master_ip = master_ip
        self.master_port = master_port
        self.num_works = num_works
        self.rank = rank

        if method == "init_weight_update_group":
            dist.init_process_group(backend="nccl" if torch.cuda.is_available(
            ) else "gloo", init_method=f"tcp://{master_ip}:{master_port}", rank=rank, world_size=num_works)
            return True
        elif method == "broadcast_weight":
            weight = torch.ones((3, 4), dtype=torch.float32)
            if torch.cuda.is_available():
                weight = weight.cuda()
            dist.broadcast(weight, src=0)
            return weight
        else:
            raise ValueError(f"method is error:{method}")


ray.init(num_cpus=8, num_gpus=8, log_to_driver=True)

num_works = 2
works = [LLMWork.remote(i) for i in range(num_works)]

work_id_refs = [work.get_work_id.remote() for work in works]
result = ray.get(work_id_refs)

print(result)

master_ip = "127.0.0.1"
master_port = 1234

handle_list = []

for index, work in enumerate(works):
    handle = work.collective_rpc.remote(
        "init_weight_update_group", master_ip, master_port, num_works, result[index])
    handle_list.append(handle)
    
init_list = ray.get(handle_list)
print(init_list)


handle_list = []
for index, work in enumerate(works):
    handle = work.collective_rpc.remote(
        "broadcast_weight", master_ip, master_port, num_works, result[index])
    handle_list.append(handle)
    
weight_list = ray.get(handle_list)
print(weight_list)

ray.shutdown()
