import ray
from ray.util import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM


class MyLLM:
    def __init__(self, model, work_extension_cls:str=None, **kwargs):
        self.model = model
        self.kwargs = kwargs
        
        if work_extension_cls:
            module_name, class_name = work_extension_cls.rsplit(".", 1)
            module = __import__(module_name, fromlist=[class_name])
            self.work_extension = getattr(module, class_name)()
        else:
            self.work_extension = None
    
    def infer(self, prompt):
        if self.work_extension:
            output = self.work_extension.pre_process(prompt)
        
        output = f"text:{prompt}, pre_process;{output}, "
        
        if self.work_extension:
            output = self.work_extension.post_process(output)
        
        return output
        

ray.init(num_cpus=8, num_gpus=4, log_to_driver=True)

pg = placement_group(bundles=[{"CPU": 4, "GPU": 2}]
                     * 2, strategy="STRICT_PACK")

pgss = PlacementGroupSchedulingStrategy(
    placement_group=pg, placement_group_bundle_index=0, placement_group_capture_child_tasks=True)

llm = ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=pgss)(MyLLM).remote("","example_utils.RLHFWorkerExtension")
result = ray.get(llm.infer.remote("Hello what is it?"))


print(result)


