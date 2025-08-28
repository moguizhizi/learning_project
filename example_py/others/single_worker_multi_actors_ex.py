import ray
import os

@ray.remote(num_cpus=0.5)
class Counter:
    def __init__(self):
        pass
    
    def get_process_id(self):
        return os.getpid()
    
@ray.remote(num_cpus=0.5)
class Other:
    def __init__(self):
        pass
    
    def get_process_id(self):
        return os.getpid()

ray.init(num_cpus=4)
        

couter1 = Counter.remote()
couter2 = Counter.remote()
couter3 = Counter.remote()

result_ref1 = couter1.get_process_id.remote()
result_ref2 = couter2.get_process_id.remote()
result_ref3 = couter3.get_process_id.remote()

result = ray.get([result_ref1, result_ref2, result_ref3])

print(result)



other1 = Other.remote()
other2 = Other.remote()
other3 = Other.remote()

result_other1 = other1.get_process_id.remote()
result_other2 = other2.get_process_id.remote()
result_other3 = other3.get_process_id.remote()

result = ray.get([result_other1, result_other2, result_other3])

print(result)

ray.shutdown()