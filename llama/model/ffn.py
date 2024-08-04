import torch
import torch.nn as nn
import torch.nn.functional as F
from ..global_args import ModelArgs, DistributedArgs
from .. import mlu as backend

import time
from ..timer import _GLOBAL_TIMER
from ..kernels import mixed_prec_matmul as mpm
from ..kernels import fused_gelu


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
        # self.dw1 = nn.Parameter(self.w1.weight[seg_beg:seg_end, :].t().contiguous(), requires_grad=False)
        # self.dw3 = nn.Parameter(self.w3.weight[seg_beg:seg_end, :].t().contiguous(), requires_grad=False)
        self.up_proj = nn.Parameter(torch.cat(
            [self.w1.weight[seg_beg:seg_end, :], self.w3.weight[seg_beg:seg_end, :]], dim=0).t().contiguous(),
        requires_grad=False)
        self.dw2 = nn.Parameter(self.w2.weight[:, seg_beg:seg_end].t().contiguous(), requires_grad=False)
        del self.w1, self.w2, self.w3
        
    def gelu(self, x: torch.Tensor):
        # # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        # tmp = x.matmul(self.dw1)
        # swished = torch.sigmoid(tmp) * tmp
        # # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        # tmpv = x.matmul(self.dw3)
        # # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim * 2)
        
        tmp = x.matmul(self.up_proj)
        
        mpm.sync();
        a = time.time()
        
        xw1, xw3 = torch.chunk(tmp, 2, dim=-1)
        swished = torch.sigmoid(xw1) * xw1
        gelud = swished * xw3
        
        mpm.sync();
        b = time.time()
        _GLOBAL_TIMER.add('GeLU', b - a)
        
        return gelud
        
        # shape = x.shape
        # x = x.reshape(-1, shape[-1])
        # y = torch.zeros((x.shape[0], x.shape[1] // 2), device=self.args.device)
        # fused_gelu.gelu(x, y)
        # y = y.reshape(*shape[:-1], x.shape[1] // 2)
        # print(y.shape)
        return y

    def forward(self, x: torch.Tensor):
        mpm.sync();
        a = time.time()
        
        x = self.gelu(x)
        x = x.matmul(self.dw2)
        
        mpm.sync();
        b = time.time()
        _GLOBAL_TIMER.add('FeedForward', b - a)
        
        x = backend.allreduce_sum(x)
        
        return x