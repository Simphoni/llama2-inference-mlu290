import math, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..global_args import ModelArgs, DistributedArgs
from typing import Tuple
from .. import mlu as backend

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: Tuple[torch.Tensor, torch.Tensor]):
    dims = freqs_complex[0].shape
    x_reshaped = x.view(-1, 2).transpose(0, 1)
    x_real = x_reshaped[0,:].contiguous().view(-1, *dims)
    x_image = x_reshaped[1,:].contiguous().view(-1, *dims)
    freqs_real = freqs_complex[0]
    freqs_image = freqs_complex[1]
    x_out_real = x_real * freqs_real - x_image * freqs_image
    x_out_image = x_real * freqs_image + x_image * freqs_real
    x_out = torch.cat([x_out_real, x_out_image], dim=-1)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs, dist_args: DistributedArgs):
        super().__init__()
        
        self.dist_args = dist_args
        
        self.n_heads_kv = args.n_heads
        self.n_heads_q = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.n_heads_tp = args.n_heads // dist_args.model_tensor_parallel_size
        # self.n_rep = self.n_heads_q // self.n_heads_kv

        self.wq = nn.Linear(args.dim, self.n_heads_q * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_heads_kv * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_heads_kv * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, 1, self.n_heads_tp, self.head_dim)).to(args.device)
        self.cache_v = torch.zeros((args.max_batch_size, 1, self.n_heads_tp, self.head_dim)).to(args.device)

    def crop_parameters(self):
        n_heads_tp = self.n_heads_tp
        head_dim = self.head_dim
        head_beg = n_heads_tp * self.dist_args.model_tensor_parallel_rank
        head_end = head_beg + n_heads_tp
        self.dwq = nn.Parameter(self.wq.weight[head_beg * head_dim : head_end * head_dim, :].clone())
        self.dwk = nn.Parameter(self.wk.weight[head_beg * head_dim : head_end * head_dim, :].clone())
        self.dwv = nn.Parameter(self.wv.weight[head_beg * head_dim : head_end * head_dim, :].clone())
        self.dwo = nn.Parameter(self.wo.weight[:, head_beg * head_dim : head_end * head_dim].clone())
        del self.wq, self.wk, self.wv, self.wo

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):        
        batch_size, seq_len, _ = x.shape  # (B, 1, Dim)

        # (B, 1, Dim) -> (B, 1, N_Head_TP * Head_Dim)
        xq = x.matmul(self.dwq.t())
        xk = x.matmul(self.dwk.t())
        xv = x.matmul(self.dwv.t())

        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_tp, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_heads_tp, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_heads_tp, self.head_dim)

        # (B, 1, H_Q, Head_Dim) --> (B, 1, H_Q, Head_Dim)
        xq = apply_rotary_embeddings(xq, freqs_complex)
        # (B, 1, H_KV, Head_Dim) --> (B, 1, H_KV, Head_Dim)
        xk = apply_rotary_embeddings(xk, freqs_complex)

        # Replace the entry in the cache
        #self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        # print(self.cache_k.shape, xk.shape)
        self.cache_k = torch.cat((self.cache_k[:batch_size, :start_pos], xk), dim=1)
        # print(self.cache_k[:batch_size, start_pos : start_pos + seq_len])
        self.cache_v = torch.cat((self.cache_v[:batch_size, :start_pos], xv), dim=1)
        #self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # (B, Seq_Len_KV, H_KV, Head_Dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.

        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        # keys = repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        # values = repeat_kv(values, self.n_rep)

        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) * (1.0 / math.sqrt(self.head_dim))
        # (B, H_Q, 1, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)
        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        output = output.matmul(self.dwo.t()) # (B, 1, Dim_TP) -> (B, 1, Dim)
        output = backend.allreduce_sum(output)
        return output
