#include <iostream>
#include <iomanip>

#include "data_generator.h"
#include "cuda_benchmark_base.h"
#include "vulkan_benchmark_base.h"

int main() {
  std::cout << "vk_radix_sort benchmark" << std::endl;

  int size = 32;
  auto keys = GenerateUniformRandomData(size);

  std::cout << "keys" << std::endl;
  for (int i = 0; i < size; i++) {
    std::cout << std::hex << std::setw(9) << keys[i];
  }
  std::cout << std::dec << std::endl;

  CudaBenchmarkBase cuda_benchmark;
  VulkanBenchmarkBase vulkan_benchmark;

  auto result = vulkan_benchmark.GlobalHistogram(keys);

  constexpr uint32_t RADIX = 256;
  std::cout << "histogram:" << std::endl;
  for (int i = 0; i < 4; i++) {
    std::cout << "pass " << i << ":" << std::endl;
    for (int j = 0; j < RADIX; j++)
      std::cout << result.histogram[i * RADIX + j] << ' ';
    std::cout << std::endl;
  }

  std::cout << "scan:" << std::dec << std::endl;
  for (int i = 0; i < 4; i++) {
    std::cout << "pass " << i << ":" << std::endl;
    for (int j = 0; j < RADIX; j++)
      std::cout << result.histogram_cumsum[i * RADIX + j] << ' ';
    std::cout << std::endl;
  }

  return 0;
}
