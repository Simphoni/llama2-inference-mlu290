import torch
import torch.nn as nn
from ..global_args import ModelArgs, DistributedArgs
from .attention import SelfAttention
from .rmsnorm import RMSNorm
from .ffn import FeedForward

class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs, dist_args: DistributedArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args, dist_args)
        self.feed_forward = FeedForward(args, dist_args)

        # Normalization BEFORE the attention block
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        # Normalization BEFORE the feed forward block
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
    
    def crop_parameter(self):
        self.attention.crop_parameters()
        self.feed_forward.crop_parameters()
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out