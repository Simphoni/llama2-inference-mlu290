import torch
import torch_mlu
import time
import torch_mlu.core.mlu_model as ct
from llama.kernels import mixed_prec_matmul, fused_gelu, fused_self_attn_decode, device_info
import transformers

torch.manual_seed(0)

ct.set_cnml_enabled(False)
ct.set_device(0)


def gemm_perf(a, b, c):
    n = 4096
    m = 4096
    k = 4096
    abtype = torch.float32
    scale = 1
    if abtype == torch.int8:
        scale = 63
    a = (scale * torch.randn((n, m))).to(abtype).to('mlu')
    b = (scale * torch.randn((k, m))).to(abtype).to('mlu')
    c = torch.zeros((n, k), dtype=torch.float32).to('mlu')

    flops, ms = mixed_prec_matmul.matmul_perf(a, b, c)
    print(flops, ms, "ms")
    print(flops / (ms * 1e-3) * 1e-12)
    print(c.cpu())

def torch_gelu(mat):
    xw1, xw3 = torch.chunk(mat, 2, dim=-1)
    swished = torch.sigmoid(xw1) * xw1
    return swished * xw3

def test_gelu(a):
    a = torch.randn((11008 // 2, 8192), dtype=torch.float32).to('mlu')
    
    n, m = a.shape
    b = torch.zeros((n, m // 2), dtype=torch.float32, device='mlu')
    fused_gelu.gelu(a, b)
    b_ans = torch_gelu(a)
    diff = (b_ans - b).abs()
    print(b.cpu())
    print(diff.cpu())
    diff /= b_ans.abs()
    print(diff.cpu())
    fused_gelu.gelu_perf(a, b)

    for i in range(32):
        torch_gelu(a)

    mixed_prec_matmul.sync()
    st = time.time()

    for i in range(64):
        torch_gelu(a)

    mixed_prec_matmul.sync()
    ed = time.time()
    print((ed - st) / 64 * 1e3, "ms")

def test_thin_mm(bs : int):
    torch_mlu._MLUC._set_quantized_bitwidth(16)
    a = torch.randn((bs, 4096)).to('mlu').to(torch.float16).to(torch.float32)
    b = torch.randn((4096, 4096)).to('mlu').to(torch.float16).to(torch.float32)
    c = torch.zeros_like(a).to(torch.float16)
    
    c_ans = torch.matmul(a, b.t())
    torch_mlu._MLUC._set_quantized_bitwidth(31)
    c_prec = torch.matmul(a, b.t())
    fused_self_attn_decode.matmul_perf(a.to(torch.float16), b.to(torch.float16), c)
    c = c.to(torch.float32)
    
    print(c.cpu())
    print(c_ans.cpu())
    print(c_prec.cpu())
    diff = (c - c_prec).abs()
    print("diff max:", diff.max().cpu())
    # print("diff cnt", (diff.cpu() > 2e-4).sum())
    diff = diff / c_prec.abs().clamp_min(1e-2)
    print("diff max rel:", diff.max().cpu())
    print('-' * 30)
    
    diff = (c_ans - c_prec).abs()
    print("diff max:", diff.max().cpu())
    # print("diff cnt", (diff.cpu() > 2e-4).sum())
    diff = diff / c_prec.abs().clamp_min(1e-2)
    print("diff max rel:", diff.max().cpu())
    print('-' * 30)
    
# device_info.print_attrs()

test_thin_mm(1)

# test_thin_mm(4)
# test_thin_mm(16)
# test_thin_mm(64)