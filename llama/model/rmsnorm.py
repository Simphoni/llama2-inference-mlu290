import torch
import torch.nn as nn
from .. import mlu as backend

import time
from ..timer import _GLOBAL_TIMER
from ..kernels import mixed_prec_matmul as mpm


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        dev = x.device
        x = x.cpu()
        res = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        res = res.to(dev)
        return res

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        mpm.sync();
        a = time.time()
        
        output = self.weight * self._norm(x.float()).type_as(x)
        
        mpm.sync();
        b = time.time()
        _GLOBAL_TIMER.add('RMSNorm', b - a)
        
        return output