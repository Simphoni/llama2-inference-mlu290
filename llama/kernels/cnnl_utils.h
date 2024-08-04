#pragma once

#include <cnnl.h>
#include <cstdio>
#include <memory>

#define cnnlSafeCall(err) __cnnlSafeCall(err, __FILE__, __LINE__)

inline void __cnnlSafeCall(cnnlStatus_t err, const char *file, const int line) {
  if (err != CNNL_STATUS_SUCCESS) {
    fprintf(stderr, "%s:%d: %s\n", file, line, cnnlGetErrorString(err));
    throw;
  }
}

template <typename T, cnnlStatus_t (*creator)(T *),
          cnnlStatus_t (*destroyer)(T)>
class WrapperBase {
private:
  T val;

public:
  WrapperBase() { cnnlSafeCall(creator(&val)); }
  ~WrapperBase() { cnnlSafeCall(destroyer(val)); }
  T &get() { return val; }

  static inline std::shared_ptr<WrapperBase<T, creator, destroyer>> build() {
    return std::shared_ptr<WrapperBase<T, creator, destroyer>>(
        new WrapperBase<T, creator, destroyer>);
  }
};

#define SPECIALIZE_WRAPPER_TYPE(name, type, prefix)                            \
  using name = WrapperBase<type, prefix##Create, prefix##Destroy>

SPECIALIZE_WRAPPER_TYPE(ContextHandle, cnnlHandle_t, cnnl);
SPECIALIZE_WRAPPER_TYPE(MatMulDesc, cnnlMatMulDescriptor_t, cnnlMatMulDesc);
SPECIALIZE_WRAPPER_TYPE(MatMulAlgo, cnnlMatMulAlgo_t, cnnlMatMulAlgo);
