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
from .model import Transformer
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
    device_id = local_rank
    dist_args = DistributedArgs(world_size, world_rank, local_rank, device_id)
    dist_args.default_group = backend.init_dev(dist_args)
    
    # make data parallel configuration
    dist_args.model_tensor_parallel_rank = world_rank
    dist_args.model_tensor_parallel_size = world_size
    
    return dist_args


class LLaMA:

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
        model_args.device = device

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()
        
        torch.set_default_dtype(torch.float32)
        
        model = Transformer(model_args, dist_args)

        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            del checkpoint['rope.freqs']
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
        return LLaMA(model, tokenizer, model_args, dist_args)

    def text_completion(self, prompts: List[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        # Convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch_size{batch_size} must be less than or equal to {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.args.max_seq_len, f"prompt_length{max_prompt_len} must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long)
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long)
        tokens = tokens.to(self.args.device)
        
        eos_reached = torch.tensor([False] * batch_size)
        prompt_tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
        for cur_pos in cur_iterator:
            gc.collect()
            print("pos=", cur_pos, ", tokens=", tokens[:, cur_pos-1:cur_pos].cpu().contiguous().flatten())
            logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
            backend.report_gpu_memory_consumption()
            if temperature > 0:
                # The temperature is applied before the softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedily select the token with the max probability
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # Only replace token if it is a padding token
            tmp = next_token.clone()
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            print(f"where op: {prompt_tokens_mask[:, cur_pos]}, {tokens[:, cur_pos]}, {tmp} = {next_token}")
            print(f"{tokens[:, cur_pos]} <- {next_token[:]}")
            #tokens[:, cur_pos] = next_token[:]
            tokens = torch.cat((tokens[:, :cur_pos], next_token.reshape(-1, 1), tokens[:,cur_pos+1:]), dim=1)
            print(tokens[:, cur_pos].cpu())
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]).cpu() & (next_token.cpu() == self.tokenizer.eos_id())
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens, out_text)
    
    def _sample_top_p(self, probs, p):
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # (B, vocab_size)
        # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
        mask = probs_sum - probs_sort > p 
        # Zero out all the probabilities of tokens that are not selected by the Top P
        probs_sort[mask] = 0.0 
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # Sample a token (its index) from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token) 
        return next_token



def inference(prompts, max_gen_len, device):
    torch.manual_seed(0)
    torch.autograd.set_grad_enabled(False)

    model = LLaMA.build(
        checkpoints_dir='../llama-2-7b-chat',
        tokenizer_path='../tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )

    out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=max_gen_len, temperature=0))
    if int(os.environ['RANK']) == 0:
        assert len(out_texts) == len(prompts)
        for i in range(len(out_texts)):
            try:
                print(f'{out_texts[i]}')
            except:
                pass
            print('-' * 50)
