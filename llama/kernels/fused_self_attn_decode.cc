#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "cnutils.h"
#include "torch_utils.h"

namespace ch = std::chrono;

namespace mlu_detail {
void self_attn_fp16(void *x, void *wq, void *xq, int bs, int nheads, int head_dim, int cluster_num);
}

void fused_self_attn(at::Tensor x, at::Tensor wq, at::Tensor wk, at::Tensor wv, at::Tensor k_cache,
                     at::Tensor v_cache, at::Tensor rotary_freqs, int nheads, int seq_len) {
  const std::string funcname = std::string(__func__);
  checkTensor(x, funcname + "::x");
  checkTensor(wq, funcname + "::wq");
  checkTensor(wk, funcname + "::wk");
  checkTensor(wv, funcname + "::wv");
  checkTensor(rotary_freqs, funcname + "::rotary_freqs");
  checkEqual(x.dim(), 3);
  checkEqual(wq.dim(), 2);
  checkEqual(wk.dim(), 2);
  checkEqual(wv.dim(), 2);
  auto a_shape = x.sizes().vec();
  int bs = a_shape[0];
  checkEqual(a_shape[1], 1);
  // assume model_hidden_dim == attention_hidden_dim
  int hidden = a_shape[2];
  int head_dim = hidden / nheads;
  checkEqual(head_dim * nheads, hidden);
  checkEqual(wq.sizes()[0], wq.sizes()[1]);
  checkEqual(wk.sizes()[0], wk.sizes()[1]);
  checkEqual(wv.sizes()[0], wv.sizes()[1]);
  checkEqual(wq.sizes()[1], hidden);
  checkEqual(wk.sizes()[1], hidden);
  checkEqual(wv.sizes()[1], hidden);
}

void manual_matmul_perf(at::Tensor x, at::Tensor wq, at::Tensor xq) {
  const std::string funcname = std::string(__func__);
  checkTensor(x, funcname + "::x");
  checkTensor(wq, funcname + "::wq");
  int bs = x.size(0);
  int m = x.size(1);

  cnrtSafeCall(cnrtSyncDevice());

  constexpr int NWARMUP = 32;
  constexpr int NPERF = 64;
  for (int i = 0; i < NPERF; i++) {
    mlu_detail::self_attn_fp16(x.data_ptr(), wq.data_ptr(), xq.data_ptr(), bs, 32, m / 32, 16);
  }
  cnrtSafeCall(cnrtSyncDevice());
  auto begin = ch::high_resolution_clock::now();
  for (int i = 0; i < NPERF; i++) {
    mlu_detail::self_attn_fp16(x.data_ptr(), wq.data_ptr(), xq.data_ptr(), bs, 32, m / 32, 16);
  }
  cnrtSafeCall(cnrtSyncDevice());
  auto end = ch::high_resolution_clock::now();
  double ms = ch::duration_cast<ch::microseconds>(end - begin).count() * 1e-3 / NPERF;
  printf("%.3lf ms, %.3lf GBps\n", ms, (1.0 * m * m * 2) / (ms * 1e-3) * 1e-9);
}

PYBIND11_MODULE(fused_self_attn_decode, m) { m.def("matmul_perf", &manual_matmul_perf); }