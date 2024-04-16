#include <iostream>
#include <iomanip>

#include "data_generator.h"
#include "cuda_benchmark_base.h"
#include "vulkan_benchmark_base.h"

int main() {
  std::cout << "vk_radix_sort benchmark" << std::endl;

  int size = 10000000;
  auto keys = GenerateUniformRandomData(size);

  std::cout << "keys" << std::endl;
  std::cout << std::hex;
  for (int i = 0; i < size && i < 16; i++) {
    std::cout << std::setfill('0') << std::setw(8) << keys[i] << " ";
  }
  std::cout << std::dec << std::endl;

  CudaBenchmarkBase cuda_benchmark;
  VulkanBenchmarkBase vulkan_benchmark;

  // histogram
  std::cout << "vulkan global histogram" << std::endl;
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

  // sort
  std::cout << "vulkan sort" << std::endl;
  result = vulkan_benchmark.Sort(keys);

  for (int i = 0; i < 4; i++) {
    std::cout << "pass " << i << ":" << std::endl;
    std::cout << std::hex;
    for (int j = 0; j < result.keys[i].size() && j < 16; j++)
      std::cout << std::setfill('0') << std::setw(8) << result.keys[i][j]
                << " ";
    std::cout << std::dec << std::endl;
  }

  bool wrong = false;
  for (int j = 1; j < result.keys[3].size(); j++) {
    if (result.keys[3][j - 1] > result.keys[3][j]) {
      wrong = true;
      break;
    }
  }

  if (wrong)
    std::cout << "wrong" << std::endl;
  else
    std::cout << "ok" << std::endl;

  return 0;
}
