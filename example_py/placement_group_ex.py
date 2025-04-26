import ray
from ray.util import placement_group

ray.init()


pg = placement_group(bundles=[{"GPU": 1, "CPU": 1}] * 4)
ray.get(pg.ready())
print(f"placement group has bundles {pg.bundle_specs=}")