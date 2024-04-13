#include <iostream>
#include <iomanip>

#include "data_generator.h"
#include "cuda_benchmark_base.h"
#include "vulkan_benchmark_base.h"

int main() {
  std::cout << "vk_radix_sort benchmark" << std::endl;

  int size = 32;
  int bits = 4;
  auto data = GenerateUniformRandomData(size, bits);

  std::cout << "data" << std::endl;
  for (int i = 0; i < size; i++) {
    std::cout << std::hex << std::setw(2) << data[i];
  }

  CudaBenchmarkBase cuda_benchmark;
  VulkanBenchmarkBase vulkan_benchmark;

  return 0;
}
