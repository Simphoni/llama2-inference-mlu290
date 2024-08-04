import torch
import torch_mlu
import torch_mlu.core.mlu_model as mlu_drv

DEVICE = 'mlu:0'

def report_diff(a_cpu: torch.Tensor, a_dev: torch.Tensor):
    diff = a_dev.cpu() - a_cpu
    diff = diff.abs() / a_cpu
    print(diff.max())

def check_bmm(ba: int, n: int, m: int, k: int):
    a = torch.randn((ba, n, k))
    b = torch.randn((ba, m, k)).transpose(1, 2)
    c_cpu = torch.bmm(a, b)
    c_dev = torch.bmm(a.to(DEVICE), b.to(DEVICE))
    report_diff(c_cpu, c_dev)

def check_mm(n: int, m: int, k: int):
    a = torch.randn((n, k))
    b = torch.randn((m, k)).transpose(0, 1)
    c_cpu = torch.mm(a, b)
    c_dev = torch.mm(a.to(DEVICE), b.to(DEVICE))
    report_diff(c_cpu, c_dev)
    
def check_silu(n: int):
    a = torch.randn(n)
    b_cpu = torch.sigmoid(a) * a
    a = a.to(DEVICE)
    b_dev = torch.sigmoid(a) * a
    report_diff(b_cpu, b_dev)

def check_softmax(n: int, m: int):
    a = torch.randn(n, m)
    b_cpu = torch.softmax(a, -1)
    a = a.to(DEVICE)
    b_dev = torch.softmax(a, -1)
    report_diff(b_cpu, b_dev)
    
def check_transpose3(n: int, m: int, k: int):
    a = torch.randn(n, m, k)
    b_cpu = a.transpose(1, 2).contiguous()
    a = a.to(DEVICE)
    b_dev = a.transpose(1, 2).contiguous()
    report_diff(b_cpu, b_dev)

def check_cat(na: int, nb: int, m: int):
    a = torch.randn(m, na)
    b = torch.randn(m, nb)
    c_cpu = torch.cat([a, b], dim=-1)
    a = a.to(DEVICE)
    b = b.to(DEVICE)
    c_dev = torch.cat([a, b], dim=-1)
    report_diff(c_cpu, c_dev)
    
    
if __name__ == "__main__":
    mlu_drv.set_device(0)
    mlu_drv.set_cnml_enabled(False)
    BMM_SHAPES = [
        [1, 32, 32, 32],
        [4, 32, 32, 32],
        [4, 16, 32, 32],
        [4, 32, 32, 128],
        [4, 32, 16, 128],
        [1, 1, 11008, 4096],
    ]
    MM_SHAPES = [
        [32, 32, 32],
        [16, 32, 32],
        [32, 32, 128],
        [32, 16, 128],
        [1, 11008, 4096],
    ]
    SILU_SHAPES = [
        32, 1024, 8192,
    ]
    SOFTMAX_SHAPES = [
        [128, 256],
        [16, 4096],
    ]
    TRANSPOSE3_SHAPES = [
        [1, 32, 4096],
        [16, 32, 128],
        [4, 512, 2],
    ]
    CAT_SHAPES = [
        [1, 1, 1024],
        [2, 2, 1024]
    ]
    print("Batch MatMul")
    for k in BMM_SHAPES:
        check_bmm(*k)
    print("MatMul")
    for k in MM_SHAPES:
        check_mm(*k)
    print("SILU")
    for k in SILU_SHAPES:
        check_silu(k)
    print("Softmax")
    for k in SOFTMAX_SHAPES:
        check_softmax(*k)
    print("Transpose3")
    for k in TRANSPOSE3_SHAPES:
        check_transpose3(*k)
    print("Cat")
    for k in CAT_SHAPES:
        check_cat(*k)