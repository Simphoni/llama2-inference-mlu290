import torch
import torch_mlu
import torch.distributed as dist
import torch_mlu.core.mlu_model as mlu_drv
import torch_mlu.distributed as mlu_dist
from .global_args import DistributedArgs

initialized = False
is_single_node = False
args_pers = None

def init_dev(args: DistributedArgs):
    global initialized, is_single_node, args_pers
    args_pers = args
    assert not initialized, "mlu.init_dev called twice"
    initialized = True
    mlu_drv.set_cnml_enabled(False)
    mlu_drv.set_device(args.device_id)
    print(f"rank {args.local_rank} using device {args.device_id}")
    if args.world_size == 1:
        is_single_node = True
        return None
    mlu_dist.init_process_group(rank=args.world_rank, world_size=args.world_size)
    return mlu_dist.get_mlu_default_group()

def get_gpu_memory_consumption():
    return (mlu_drv.memory_allocated(), mlu_drv.memory_cached())

def get_max_gpu_memory_consumption():
    return (mlu_drv.max_memory_allocated(), mlu_drv.max_memory_cached())

def report_gpu_memory_consumption():
    a, b = get_gpu_memory_consumption()
    print(f"rank {args_pers.world_rank}: alloced: {a / (1024 ** 3):.3f} GiB, cached {b / 1024 ** 3:.3f} GiB")
    a, b = get_max_gpu_memory_consumption()
    print(f"rank {args_pers.world_rank}: max alloced: {a / (1024 ** 3):.3f} GiB, max cached {b / 1024 ** 3:.3f} GiB")

def manual_clean():
    mlu_drv.empty_cached_memory()

def get_device():
    global initialized
    assert initialized, "calling mlu.get_device before mlu.init_dev (device not set)"
    return mlu_drv.mlu_device()

def allreduce_sum(x: torch.Tensor):
    global initialized
    assert initialized, "calling mlu.get_device before mlu.init_dev (device not set)"
    dev = x.device
    if not is_single_node:
        tmp = x.to(mlu_drv.mlu_device())
        dist.all_reduce(tmp, dist.ReduceOp.SUM, group=mlu_dist.get_mlu_default_group())
        x = tmp.to(dev)
        return x