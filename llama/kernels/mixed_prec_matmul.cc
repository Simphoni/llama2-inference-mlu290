#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <chrono>
#include <cstdint>
#include <vector>

#include "cnutils.h"
#include "torch_utils.h"

namespace ch = std::chrono;

cnnlHandle_t getCNNLHandle() {
  static bool initialized = false;
  static cnnlHandle_t handle;
  if (!initialized) {
    cnnlSafeCall(cnnlCreate(&handle));
  }
  return handle;
}

std::shared_ptr<TensorDesc>
getTensorDesc(const std::vector<int> &shape, cnnlDataType_t dtype,
              cnnlDataType_t onchip_dtype = CNNL_DTYPE_INVALID) {
  if (onchip_dtype == CNNL_DTYPE_INVALID) {
    onchip_dtype = dtype;
  }
  auto desc = TensorDesc::build();
  cnnlSafeCall(cnnlSetTensorDescriptor(desc->get(), CNNL_LAYOUT_NHWC, dtype,
                                       shape.size(), shape.data()));
  cnnlSafeCall(
      cnnlSetTensorDescriptorOnchipDataType(desc->get(), onchip_dtype));
  return desc;
}

std::shared_ptr<MatMulDesc> getMatMulDesc(cnnlDataType_t compute_dtype,
                                          int transa, int transb) {
  auto mm = MatMulDesc::build();
  cnnlSafeCall(cnnlSetMatMulDescAttr(mm->get(), CNNL_MATMUL_DESC_COMPUTE_TYPE,
                                     &compute_dtype, sizeof(compute_dtype)));
  cnnlSafeCall(cnnlSetMatMulDescAttr(mm->get(), CNNL_MATMUL_DESC_TRANSA,
                                     &transa, sizeof(transa)));
  cnnlSafeCall(cnnlSetMatMulDescAttr(mm->get(), CNNL_MATMUL_DESC_TRANSB,
                                     &transb, sizeof(transb)));
  return mm;
}

std::shared_ptr<MLUBuffer>
getMatMulQuantizeParam(cnnlTensorDescriptor_t input_desc, void *input,
                       int bitwidth,
                       cnnlQuantizeMode_t mode = CNNL_QUANTIZE_POSITION) {
  assert(mode == CNNL_QUANTIZE_POSITION);
  auto handle = getCNNLHandle();
  size_t workspace_size;
  cnnlSafeCall(
      cnnlGetQuantizeParamWorkspaceSize(handle, input_desc, &workspace_size));
  auto ws = MLUBuffer::build(workspace_size);
  auto position = MLUBuffer::build(3 * 4);
  cnnlSafeCall(cnnlQuantizeParam(handle, mode, input_desc, input, bitwidth,
                                 ws->get(), workspace_size, position->get(),
                                 nullptr, nullptr));
  return position;
}

// [n, m][k, m]->[n, k]
void matmul_quant_fp32_fp32_fp32(void *a, void *b, void *c, int64_t n,
                                 int64_t m, int64_t k) {
  constexpr int width = 8;
  cnnlDataType_t dtype = CNNL_DTYPE_INVALID;
  if (width == 8) {
    dtype = CNNL_DTYPE_INT8;
  } else if (width == 16) {
    dtype = CNNL_DTYPE_INT16;
  } else if (width == 31) {
    dtype = CNNL_DTYPE_INT31;
  } else {
    throw std::logic_error("bitwidth should be 8, 16, 31");
  }
  checkSafelyConvertToInt32(n);
  checkSafelyConvertToInt32(m);
  checkSafelyConvertToInt32(k);
  auto handle = getCNNLHandle();
  auto a_desc = getTensorDesc({(int)n, (int)m}, CNNL_DTYPE_FLOAT, dtype);
  auto b_desc = getTensorDesc({(int)k, (int)m}, CNNL_DTYPE_FLOAT, dtype);
  auto c_desc =
      getTensorDesc({(int)n, (int)k}, CNNL_DTYPE_FLOAT, CNNL_DTYPE_FLOAT);
  std::shared_ptr<MatMulDesc> mm = getMatMulDesc(CNNL_DTYPE_FLOAT, false, true);
  std::shared_ptr<MatMulAlgo> algo = MatMulAlgo::build();
  float alpha = 1.0, beta = 0;
  cnnlSafeCall(cnnlGetQuantizeMatMulAlgorithm(
      handle, mm->get(), a_desc->get(), b_desc->get(), c_desc->get(),
      CNNL_MATMUL_FASTEST, &algo->get()));
  auto a_quant = getMatMulQuantizeParam(a_desc->get(), a, width);
  auto b_quant = getMatMulQuantizeParam(b_desc->get(), b, width);
  size_t ws_size = 0;
  cnnlSafeCall(cnnlGetQuantizeMatMulWorkspaceSize(
      handle, mm->get(), a_desc->get(), b_desc->get(), c_desc->get(),
      algo->get(), &ws_size));
  auto ws = MLUBuffer::build(ws_size);
  cnnlSafeCall(cnnlQuantizeMatMul(
      handle, mm->get(), &alpha, a_desc->get(), a, a_quant->get(), nullptr,
      nullptr, b_desc->get(), b, b_quant->get(), nullptr, nullptr, &beta,
      c_desc->get(), c, algo->get(), ws->get(), ws_size));
  cnrtSafeCall(cnrtSyncDevice());
}

void matmul_i8_i8_fp32(void *a, void *b, void *c, int64_t n, int64_t m,
                       int64_t k) {
  checkSafelyConvertToInt32(n);
  checkSafelyConvertToInt32(m);
  checkSafelyConvertToInt32(k);
  auto handle = getCNNLHandle();
  auto a_desc = getTensorDesc({(int)n, (int)m}, CNNL_DTYPE_INT8);
  auto b_desc = getTensorDesc({(int)k, (int)m}, CNNL_DTYPE_INT8);
  auto c_desc =
      getTensorDesc({(int)n, (int)k}, CNNL_DTYPE_FLOAT, CNNL_DTYPE_INT31);
  float alpha = 1.0, beta = 0;
  cnnlSafeCall(cnnlMatMul(handle, false, true, &alpha, a_desc->get(), a,
                          b_desc->get(), b, &beta, c_desc->get(), &c));
}

void matmul(at::Tensor a, at::Tensor b, at::Tensor output) {
  checkTensor(a, "matmul::a");
  checkTensor(b, "matmul::b");
  checkTensor(output, "matmul::output");
  checkEqual(b.dim(), 2);
  checkEqual(a.dim(), output.dim());
  auto a_shape = a.sizes().vec();
  auto b_shape = b.sizes().vec();
  auto output_shape = output.sizes().vec();
  int64_t ta = 1;
  for (int i = 0; i + 1 < (int)a_shape.size(); ++i) {
    checkEqual(a_shape[i], output_shape[i]);
    ta *= a_shape[i];
  }
  checkEqual(a_shape.back(), b_shape.back());
  checkEqual(b_shape.front(), output_shape.back());
  if (tensorTypeIs<float>(a) && tensorTypeIs<float>(b) &&
      tensorTypeIs<float>(output)) {
    matmul_quant_fp32_fp32_fp32(a.data_ptr(), b.data_ptr(), output.data_ptr(),
                                ta, a_shape.back(), output_shape.back());
  } else {
    throw std::logic_error("MatMul not implemented for provided data type");
  }
}

std::pair<double, double> matmul_perf(at::Tensor a, at::Tensor b,
                                      at::Tensor output) {
  constexpr int NWARMUP = 16;
  constexpr int NPERF = 32;
  checkTensor(a, "matmul::a");
  checkTensor(b, "matmul::b");
  checkTensor(output, "matmul::output");
  checkEqual(b.dim(), 2);
  checkEqual(a.dim(), output.dim());
  auto a_shape = a.sizes().vec();
  auto b_shape = b.sizes().vec();
  auto output_shape = output.sizes().vec();
  int64_t ta = 1;
  for (int i = 0; i + 1 < (int)a_shape.size(); ++i) {
    checkEqual(a_shape[i], output_shape[i]);
    ta *= a_shape[i];
  }
  checkEqual(a_shape.back(), b_shape.back());
  checkEqual(b_shape.front(), output_shape.back());
  int64_t n = ta;
  int64_t m = a_shape.back();
  int64_t k = output_shape.back();

  auto handle = getCNNLHandle();

#define GEMM_ARGS a.data_ptr(), b.data_ptr(), output.data_ptr(), n, m, k
  auto run_test = [&]() {
    if (tensorTypeIs<float>(a) && tensorTypeIs<float>(b) &&
        tensorTypeIs<float>(output)) {
      matmul_quant_fp32_fp32_fp32(GEMM_ARGS);
    } else if (tensorTypeIs<int8_t>(a) && tensorTypeIs<int8_t>(b) &&
               tensorTypeIs<float>(output)) {
      matmul_i8_i8_fp32(GEMM_ARGS);
    } else {
      throw std::logic_error("MatMul not implemented for provided data type");
    }
  };
#undef GEMM_ARGS
  for (int i = 0; i < NWARMUP; i++) {
    run_test();
  }
  cnrtSafeCall(cnrtSyncDevice());
  auto begin = ch::high_resolution_clock::now();
  for (int i = 0; i < NPERF; i++) {
    run_test();
  }
  cnrtSafeCall(cnrtSyncDevice());
  auto end = ch::high_resolution_clock::now();
  double ret = ch::duration_cast<ch::microseconds>(end - begin).count() * 1e-3;
  double flops = 2.0 * n * m * k;
  fprintf(stderr, "%.10lf\n", ret);
  return std::make_pair(flops, ret / NPERF);
}

void manual_sync() { cnrtSafeCall(cnrtSyncDevice()); }

PYBIND11_MODULE(mixed_prec_matmul, m) {
  m.def("matmul", &matmul,
        "MatMul forward (MLU), requires matrix B transposed.");
  m.def("matmul_perf", &matmul_perf,
        "MatMul forward (MLU) performance measurement, requires matrix B "
        "transposed.");
  m.def("sync", &manual_sync, "manually sync device");
}