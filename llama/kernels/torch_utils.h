#include <string>
#include <torch/extension.h>

inline void checkTensorDevice(at::Tensor a, const std::string &name) {
  if (a.device().type() != c10::DeviceType::MLU) {
    fprintf(stderr, "[ERROR] tensor %s expected MLU, got %s\n", name.data(),
            DeviceTypeName(a.device().type()).data());
    throw;
  }
}

inline void checkTensorContiguous(at::Tensor a, const std::string &name) {
  if (!a.is_contiguous()) {
    fprintf(stderr, "[ERROR] tensor %s not contiguous\n", name.data());
    throw;
  }
}

inline void checkTensor(at::Tensor a, const std::string &name) {
  checkTensorDevice(a, name);
  checkTensorContiguous(a, name);
}

inline void checkEqualWithDbg(int64_t a, int64_t b, int line) {
  if (a != b) {
    fprintf(stderr, "%s:%d [ERROR] checkEqual failed (%ld != %ld)\n", __FILE__,
            line, a, b);
    throw;
  }
}

inline bool checkSafelyConvertToInt32(int64_t a) {
  return a >= 0 && a <= INT_MAX;
}

#define checkEqual(a, b) checkEqualWithDbg(a, b, __LINE__)

template <typename T> inline bool tensorTypeIs(const at::Tensor &a) {
  return a.dtype().id() == caffe2::TypeIdentifier::Get<T>();
}