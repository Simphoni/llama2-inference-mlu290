import torch
import torch.nn as nn
from ..global_args import ModelArgs, DistributedArgs
from .rmsnorm import RMSNorm
from .encoder import EncoderBlock
from .. import mlu as backend

import time
from ..timer import _GLOBAL_TIMER
from ..kernels import mixed_prec_matmul as mpm


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0, f"head_dim({head_dim}) % 2 != 0"
    theta_numerator = torch.arange(0, head_dim, 2, dtype=torch.float64)
    thetas = (torch.full([head_dim // 2], theta) ** (theta_numerator / head_dim)).reciprocal()
    m = torch.arange(seq_len)
    zero_mat = torch.zeros((seq_len, head_dim // 2), dtype=torch.float64)
    thetas = zero_mat + thetas.view(1, head_dim // 2)
    m = zero_mat + m.view(seq_len, 1)
    freqs = torch.mul(m, thetas)
    freqs_complex = (torch.cos(freqs).to(torch.float32).to(device), torch.sin(freqs).to(torch.float32).to(device))
    return freqs_complex


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, dist_args: DistributedArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.dist_args = dist_args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args, dist_args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)
    
    def crop_parameter(self):
        for encoder in self.layers:
            encoder.crop_parameter()

    def forward(self, tokens: torch.Tensor, start_pos: int):
        mpm.sync()
        a = time.time()
        
        # (B, Seq_Len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        #freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        freqs_complex = (
            self.freqs_complex[0][start_pos:start_pos + seq_len, :],
            self.freqs_complex[1][start_pos:start_pos + seq_len, :]
        )
        
        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        
        mpm.sync()
        b = time.time()
        
        _GLOBAL_TIMER.add('Transformer', b - a)
        
        return output