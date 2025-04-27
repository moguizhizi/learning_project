import ray
import os

@ray.remote
class Counter:
    def __init__(self):
        pass
    
    def get_process_id(self):
        return os.getgid()

ray.init()
        

couter1 = Counter.remote()
couter2 = Counter.remote()
couter3 = Counter.remote()

result_ref1 = couter1.get_process_id.remote()
result_ref2 = couter2.get_process_id.remote()
result_ref3 = couter3.get_process_id.remote()

result = ray.get([result_ref1, result_ref2, result_ref3])

print(result)

ray.shutdown()