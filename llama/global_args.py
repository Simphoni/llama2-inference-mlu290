import os
from dataclasses import dataclass

@dataclass
class ModelArgs:
    dim: int = 4096
    ffn_hidden_size: int = 11008
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = -1
    multiple_of: int = 256
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 128

    device: str = "cpu"
    
@dataclass
class DistributedArgs:
    world_size: int = 1
    world_rank: int = 0
    local_rank: int = 0
    device_id: int = 0
    default_group = None
    
    model_tensor_parallel_size: int = 1
    model_tensor_parallel_rank: int = 0
    
    def is_rank_0(self) -> bool:
        return self.world_rank == 0

    def is_rank_last(self) -> bool:
        return self.world_rank + 1 == self.world_size