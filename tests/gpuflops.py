import torch
import torch_mlu
import torch_mlu.core.mlu_model as ct
from torch_mlu.core.device.notifier import Notifier

PERF_ROUNDS = 32

def flops_test(n: int, m: int, k: int):
    dev = ct.mlu_device()
    a = torch.zeros((n, k), dtype=torch.float32).to(dev)
    b = torch.zeros((k, m), dtype=torch.float32).to(dev)
    c = torch.zeros((n, m), dtype=torch.float32).to(dev)

    for _ in range(PERF_ROUNDS):
        torch.matmul(a, b, out=c)
    st = Notifier()
    ed = Notifier()
    st.place()
    for _ in range(PERF_ROUNDS):
        torch.matmul(a, b, out=c)
    ed.place()
    st.synchronize()
    ed.synchronize()
    elapsed = st.elapsed_time(ed) / PERF_ROUNDS * 1e-6;
    print(f"n={n}, m={m}, k={k}, ave={elapsed * 1000} ms, TFLOPS={n * m * k * 2 / elapsed * 1e-12}, GBps={(n * m + n * k + m * k) * 4 / elapsed * 1e-9}")

ct.set_cnml_enabled(False)
ct.set_device(int(4))

# torch_mlu._MLUC._set_quantized_bitwidth(16)
flops_test(1, 4096, 4096)
flops_test(4, 4096, 4096)
flops_test(16, 4096, 4096)
flops_test(64, 4096, 4096)
# flops_test(4, 4096 * 3, 4096)
# flops_test(16, 4096 * 3, 4096)
# flops_test(64, 4096 * 3, 4096)
# flops_test(4, 11008, 4096)