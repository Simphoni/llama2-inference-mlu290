import torch
import time
import torch_mlu
import torch.nn as nn
import torch_mlu.core.mlu_model as ct
from sentencepiece import SentencePieceProcessor
from pathlib import Path

def test_checkpoint():
    tokenizer = SentencePieceProcessor()
    tokenizer.load("./tokenizer.model")
    print(tokenizer.vocab_size())
    print(tokenizer.pad_id())
    print(tokenizer.eos_id())
    checkpoints_dir = "./llama-2-7b-chat"
    ckpt_path = checkpoints_dir + "/weights.pth"
    checkpoint = torch.load(ckpt_path)
    for k in checkpoint:
        print(k, checkpoint[k].shape)
    
def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0, f"head_dim({head_dim}) % 2 != 0"
    theta_numerator = torch.arange(0, head_dim, 2, dtype=torch.float64)
    thetas = 1.0 / (theta ** (theta_numerator / head_dim))
    m = torch.arange(seq_len)
    zero_mat = torch.zeros((seq_len, head_dim // 2), dtype=torch.float64)
    thetas = zero_mat + thetas.view(1, head_dim // 2)
    m = zero_mat + m.view(seq_len, 1)
    freqs = torch.mul(m, thetas)
    freqs_complex = (torch.cos(freqs).to(torch.float16).to(device), torch.sin(freqs).to(torch.float16).to(device))
    return freqs_complex

def test_precomp_freq():
    a = precompute_theta_pos_frequencies(4096 // 32, 1024, ct.mlu_device())
    print(a[0].cpu(), a[1].cpu())
    print(a[0].dtype, a[0].shape)

def test_mm(ba, n, m, k):
    a = torch.randn(ba, n, k).to(torch.float32).to(ct.mlu_device())
    b = torch.randn(ba, m, k).to(torch.float32).to(ct.mlu_device())
    c = torch.bmm(a, b.transpose(1, 2))

class mymod(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw = torch.zeros(100)
        self.param = nn.Parameter(torch.zeros(100))
    
    def print(self):
        print("raw:", self.raw.device)
        print("param:", self.param.device)

ct.set_device(0)
ct.set_cnml_enabled(False)
ct.set_device(0)
ct.set_cnml_enabled(False)

# test_mm(16, 32, 128, 64)
# test_checkpoint()

torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)

with torch.no_grad():
    a = torch.ones((2,5), dtype=torch.float32).to('mlu')
    b = torch.ones((4), dtype=torch.float32).to('mlu') * 2
    print(a.cpu())
    print(b.cpu())

    # print(a.device, b.device)
    # print(a.cpu())
    # a[0] = b
    a[1,0:4] = b
    print(a.cpu())
    # print(a)

    # a = a.cpu()
    # b = b.cpu()
    # index = index.cpu()
    # print(a.cpu())
