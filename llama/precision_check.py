import time, os, gc
import json
import psutil
from pathlib import Path
from typing import Optional, List

import torch
import torch_mlu
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from .global_args import DistributedArgs, ModelArgs
from .model import DummyTransformer as Transformer
from . import mlu as backend

def get_distributed_args() -> DistributedArgs:
    world_size = 1 
    world_rank = 0
    local_rank = 0
    env = list(dict(os.environ).keys())
    if not "WORLD_SIZE" in env or not "RANK" in env or not "LOCAL_RANK" in env:
        print(f"{__file__}:{__name__} [WARING] distributed args not found in os.environ, using single node impl")
    else:
        world_size = int(os.environ.get("WORLD_SIZE", default=1))
        world_rank = int(os.environ.get("RANK", default=0))
        local_rank = int(os.environ.get("LOCAL_RANK", default=0))
        assert world_rank < world_size, f"world_rank({world_rank}) < world_size({world_size}) check failed"
    device_id = local_rank + 4
    dist_args = DistributedArgs(world_size, world_rank, local_rank, device_id)
    dist_args.default_group = backend.init_dev(dist_args)
    
    # make data parallel configuration
    dist_args.model_tensor_parallel_rank = world_rank
    dist_args.model_tensor_parallel_size = world_size
    
    return dist_args


class DummyLLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs, dist_args: DistributedArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args
        self.dist_args = dist_args
        # check argument validity
        tp_size = dist_args.model_tensor_parallel_size
        n_head = model_args.n_heads
        ffn_hidden_size = model_args.ffn_hidden_size
        assert n_head % tp_size == 0, f"n_head({n_head}) % tp_size({tp_size}) != 0"
        assert ffn_hidden_size % tp_size == 0, f"ffn_hidden_size({ffn_hidden_size}) % tp_size({tp_size}) != 0"

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        dist_args = get_distributed_args()

        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()
        
        torch.set_default_dtype(torch.float32)
        
        model = Transformer(model_args, dist_args)

        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            del checkpoint['rope.freqs']
            for i in range(1, 32):
                del checkpoint[f"layers.{i}.attention.wq.weight"]
                del checkpoint[f"layers.{i}.attention.wk.weight"]
                del checkpoint[f"layers.{i}.attention.wv.weight"]
                del checkpoint[f"layers.{i}.attention.wo.weight"]
                del checkpoint[f"layers.{i}.feed_forward.w1.weight"]
                del checkpoint[f"layers.{i}.feed_forward.w2.weight"]
                del checkpoint[f"layers.{i}.feed_forward.w3.weight"]
                del checkpoint[f"layers.{i}.attention_norm.weight"]
                del checkpoint[f"layers.{i}.ffn_norm.weight"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")
        del checkpoint
        gc.collect()
            
        model.crop_parameter()
        print("done cropping parameters")
        model = model.to(model_args.device)
        if False and dist_args.world_rank == 0:
            for param in model.parameters():
                print(type(param.data), param.device, param.size())
        backend.report_gpu_memory_consumption()
        print(psutil.virtual_memory()._asdict())
        return DummyLLaMA(model, tokenizer, model_args, dist_args)

    def text_completion(self, prompts: List[str], cur_pos: int):
        prompt_tokens = [self.tokenizer.Encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        batch_size = len(prompt_tokens)
        total_len = 64

        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long)
        tokens = tokens.to(self.args.device)
        
        return self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
    



def inference(prompts, max_gen_len, device):
    torch.manual_seed(0)
    torch.autograd.set_grad_enabled(False)

    model_mlu = DummyLLaMA.build(
        checkpoints_dir='../llama-2-7b-chat',
        tokenizer_path='../tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device='mlu'
    )
    model_cpu = DummyLLaMA.build(
        checkpoints_dir='../llama-2-7b-chat',
        tokenizer_path='../tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device='cpu'
    )
    
    for cur_pos in range(1, 8):
        logits_mlu = model_mlu.text_completion(prompts, cur_pos)
        logits_cpu = model_cpu.text_completion(prompts, cur_pos)
        diff = (logits_mlu.cpu() - logits_cpu).abs() / logits_cpu.abs()
        if int(os.environ['RANK']) == 0:
            print(cur_pos, ":", diff.max())
        
