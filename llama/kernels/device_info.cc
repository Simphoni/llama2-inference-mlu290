#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "cnutils.h"
#include "torch_utils.h"

void print_attr(CNdevice_attribute attr, std::string name, std::string suffix = "") {
  int data = 0;
  cnSafeCall(cnDeviceGetAttribute(&data, attr, 0));
  printf("%s = %d %s\n", name.c_str(), data, suffix.c_str());
}

void print_all_attrs() {
  print_attr(CN_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_TOTAL_SIZE, "Global Mem Size", "MiB");
  print_attr(CN_DEVICE_ATTRIBUTE_MAX_L2_CACHE_SIZE, "L2 Cache size", "B");
  print_attr(CN_DEVICE_ATTRIBUTE_MAX_CLUSTER_COUNT, "Cluster count");
  print_attr(CN_DEVICE_ATTRIBUTE_MAX_SHARED_RAM_SIZE_PER_CLUSTER, "Shared Mem Size", "B");
  print_attr(CN_DEVICE_ATTRIBUTE_NEURAL_RAM_SIZE_PER_CORE, "Nram Size", "B");
}

PYBIND11_MODULE(device_info, m) { m.def("print_attrs", &print_all_attrs); }