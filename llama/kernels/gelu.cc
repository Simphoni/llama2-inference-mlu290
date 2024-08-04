#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "cnutils.h"
#include "torch_utils.h"

namespace ch = std::chrono;

namespace mlu_detail {
void gelu_fp32(float *src, float *dst, int n, int m, int cluster_num);
}

// xw1 and xw3 are concatenated at dim0
void gelu_cat_dim0(at::Tensor src, at::Tensor dst) {
  int cluster_num;
  cnSafeCall(cnDeviceGetAttribute(&cluster_num,
                                  CN_DEVICE_ATTRIBUTE_MAX_CLUSTER_COUNT, 0));
  checkTensor(src, "gelu_cat_dim0::mat");
  checkEqual(src.dim(), 2);
  checkEqual(dst.dim(), 2);
  auto src_shape = src.sizes().vec();
  auto dst_shape = dst.sizes().vec();
  int64_t n = src_shape[0], m = src_shape[1];
  checkEqual(n, dst_shape[0]);
  checkEqual(m / 2, dst_shape[1]);
  if (tensorTypeIs<float>(src) && tensorTypeIs<float>(dst)) {
    mlu_detail::gelu_fp32(src.data_ptr<float>(), dst.data_ptr<float>(), n, m,
                          cluster_num);
  }
  cnrtSafeCall(cnrtSyncDevice());
}

void gelu_cat_dim0_perf(at::Tensor src, at::Tensor dst) {
  constexpr int NWARMUP = 32;
  constexpr int NPERF = 64;
  int cluster_num;
  cnSafeCall(cnDeviceGetAttribute(&cluster_num,
                                  CN_DEVICE_ATTRIBUTE_MAX_CLUSTER_COUNT, 0));
  checkTensor(src, "gelu_cat_dim0::mat");
  checkEqual(src.dim(), 2);
  checkEqual(dst.dim(), 2);
  auto src_shape = src.sizes().vec();
  auto dst_shape = dst.sizes().vec();
  int64_t n = src_shape[0], m = src_shape[1];
  checkEqual(n, dst_shape[0]);
  checkEqual(m / 2, dst_shape[1]);
#define GELU_ARGS                                                              \
  src.data_ptr<float>(), dst.data_ptr<float>(), n, m, cluster_num
  auto run_test = [&]() {
    if (tensorTypeIs<float>(src) && tensorTypeIs<float>(dst)) {
      mlu_detail::gelu_fp32(GELU_ARGS);
    }
  };
#undef GELU_ARGS
  for (int i = 0; i < NPERF; i++) {
    run_test();
  }
  cnrtSafeCall(cnrtSyncDevice());
  auto begin = ch::high_resolution_clock::now();
  for (int i = 0; i < NPERF; i++) {
    run_test();
  }
  cnrtSafeCall(cnrtSyncDevice());
  auto end = ch::high_resolution_clock::now();
  double ms =
      ch::duration_cast<ch::microseconds>(end - begin).count() * 1e-3 / NPERF;
  double bytes = n * m * 1.5 * sizeof(float);
  printf("%.3lf GBps\n", bytes / (ms * 1e-3) * 1e-9);
  printf("%.10lf ms\n", ms);
}

PYBIND11_MODULE(fused_gelu, m) {
  m.def("gelu", &gelu_cat_dim0);
  m.def("gelu_perf", &gelu_cat_dim0_perf);
}