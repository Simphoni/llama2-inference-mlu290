import torch
import torch.nn as nn
import torch.nn.functional as F
from ..global_args import ModelArgs, DistributedArgs
from .. import mlu as backend


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs, dist_args: DistributedArgs):
        super().__init__()
        
        self.args = args
        self.dist_args = dist_args

        hidden_dim = args.ffn_hidden_size

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def crop_parameters(self):
        args = self.args
        dist_args = self.dist_args
        seg_len = args.ffn_hidden_size // dist_args.model_tensor_parallel_size
        seg_beg = seg_len * dist_args.model_tensor_parallel_rank
        seg_end = seg_beg + seg_len
        self.dw1 = nn.Parameter(self.w1.weight[seg_beg:seg_end, :])
        self.dw3 = nn.Parameter(self.w3.weight[seg_beg:seg_end, :])
        self.dw2 = nn.Parameter(self.w2.weight[:, seg_beg:seg_end])
        del self.w1, self.w2, self.w3
        
    def gelu(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        tmp = x.matmul(self.dw1.t())
        swished = torch.sigmoid(tmp) * tmp
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        tmpv = x.matmul(self.dw3.t())
        return swished * tmpv

    def forward(self, x: torch.Tensor):
        x = self.gelu(x)
        x = x.matmul(self.dw2.t())
        x = backend.allreduce_sum(x)
        return x