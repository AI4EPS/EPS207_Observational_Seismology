import torch
import torch.distributed as dist

dist.init_process_group(backend="gloo")
print(f"I am worker {dist.get_rank()} of {dist.get_world_size()}!")

a = torch.tensor([dist.get_rank()])
dist.all_reduce(a)
print(f"all_reduce output = {a}")
