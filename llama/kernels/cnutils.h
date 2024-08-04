#pragma once

#include <cn_api.h>
#include <cnnl.h>
#include <cnrt.h>
#include <cstdio>
#include <memory>

constexpr int MAX_NRAM_BYTES = 480000;
constexpr int MAX_SRAM_BYTES = 2000000;

#define cnSafeCall(err) __cnSafeCall(err, __FILE__, __LINE__)
#define cnnlSafeCall(err) __cnnlSafeCall(err, __FILE__, __LINE__)
#define cnrtSafeCall(err) __cnrtSafeCall(err, __FILE__, __LINE__)

inline void __cnSafeCall(CNresult err, const char *file, const int line) {
  static char errstr[64];
  if (err != CN_SUCCESS) {
    const char *p = errstr;
    cnGetErrorString(err, &p);
    fprintf(stderr, "%s:%d: %s\n", file, line, p);
    throw;
  }
}

inline void __cnnlSafeCall(cnnlStatus_t err, const char *file, const int line) {
  if (err != CNNL_STATUS_SUCCESS) {
    fprintf(stderr, "%s:%d: %s\n", file, line, cnnlGetErrorString(err));
    throw;
  }
}

inline void __cnrtSafeCall(cnrtRet_t err, const char *file, const int line) {
  if (err != CNRT_RET_SUCCESS) {
    fprintf(stderr, "%s:%d: %s\n", file, line, cnrtGetErrorStr(err));
    throw;
  }
  return;
}

template <typename T, cnnlStatus_t (*create)(T *), cnnlStatus_t (*destroyer)(T)>
class CNNLWrapperBase {
private:
  T val;
  CNNLWrapperBase() { cnnlSafeCall(create(&val)); }

public:
  static inline auto build() {
    return std::shared_ptr<CNNLWrapperBase<T, create, destroyer>>(
        new CNNLWrapperBase<T, create, destroyer>);
  }
  ~CNNLWrapperBase() { cnnlSafeCall(destroyer(val)); }
  T &get() noexcept { return val; }
};

#define SPECIALIZE_CNNL_WRAPPER_TYPE(name, type, prefix)                                           \
  using name = CNNLWrapperBase<type, prefix##Create, prefix##Destroy>
#define SPECIALIZE_CNNL_WRAPPER_TYPE_VARIANT(name, type, suffix)                                   \
  using name = CNNLWrapperBase<type, cnnl##Create##suffix, cnnl##Destroy##suffix>

SPECIALIZE_CNNL_WRAPPER_TYPE(CNNLHandle, cnnlHandle_t, cnnl);
SPECIALIZE_CNNL_WRAPPER_TYPE(MatMulDesc, cnnlMatMulDescriptor_t, cnnlMatMulDesc);
SPECIALIZE_CNNL_WRAPPER_TYPE(MatMulAlgo, cnnlMatMulAlgo_t, cnnlMatMulAlgo);
SPECIALIZE_CNNL_WRAPPER_TYPE_VARIANT(TensorDesc, cnnlTensorDescriptor_t, TensorDescriptor);

// DO NOT use queue and notifier on MLU290 with CNRT==4.10.1 ddf6202
// the driver apis seem buggy

class CNRTQueue {
private:
  cnrtQueue_t val;
  CNRTQueue() { cnrtSafeCall(cnrtCreateQueue(&val)); }

public:
  static inline auto build() { return std::shared_ptr<CNRTQueue>(new CNRTQueue); }
  ~CNRTQueue() { cnrtSafeCall(cnrtDestroyQueue(val)); }
  cnrtQueue_t &get() noexcept { return val; }
  void sync() { cnrtSafeCall(cnrtSyncQueue(val)); }
};

class CNRTNotifier {
private:
  cnrtNotifier_t val;
  CNRTNotifier() { cnrtSafeCall(cnrtCreateNotifier(&val)); }

public:
  static inline auto build() { return std::shared_ptr<CNRTNotifier>(new CNRTNotifier); }
  ~CNRTNotifier() { cnrtSafeCall(cnrtDestroyNotifier(&val)); }
  cnrtNotifier_t &get() noexcept { return val; }
  void place(std::shared_ptr<CNRTQueue> q) { cnrtSafeCall(cnrtPlaceNotifier(val, q->get())); }
  void sync() { cnrtSafeCall(cnrtWaitNotifier(val)); }
  float elapsed_ms(std::shared_ptr<CNRTNotifier> end) {
    float us = 0;
    cnrtSafeCall(cnrtNotifierDuration(val, end->get(), &us));
    return us * 1e-3;
  }
};

class MLUBuffer {
private:
  void *ptr{nullptr};
  MLUBuffer(size_t size) {
    if (size > 0) {
      cnrtSafeCall(cnrtMalloc(&ptr, size));
    }
  }

public:
  static inline std::shared_ptr<MLUBuffer> build(size_t size) {
    return std::shared_ptr<MLUBuffer>(new MLUBuffer(size));
  }
  ~MLUBuffer() { cnrtSafeCall(cnrtFree(ptr)); }
  void *get() const { return ptr; }
};