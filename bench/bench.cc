#include <iostream>
#include <iomanip>

#include "data_generator.h"
#include "cpu_benchmark_base.h"
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

  CpuBenchmarkBase cpu_benchmark;
  VulkanBenchmarkBase vulkan_benchmark;

  std::cout << "================ sort ================" << std::endl;
  std::cout << "cpu sort" << std::endl;
  auto result0 = cpu_benchmark.Sort(keys);
  std::cout << "vulkan sort" << std::endl;
  auto result1 = vulkan_benchmark.Sort(keys);

  std::cout << "total time: " << static_cast<double>(result1.total_time) / 1e6
            << "ms" << std::endl;
  std::cout << "histogram time: "
            << static_cast<double>(result1.histogram_time) / 1e6 << "ms"
            << std::endl;
  std::cout << "scan time: " << static_cast<double>(result1.scan_time) / 1e6
            << "ms" << std::endl;
  std::cout << "binning0 time: "
            << static_cast<double>(result1.binning_times[0]) / 1e6 << "ms"
            << std::endl;
  std::cout << "binning1 time: "
            << static_cast<double>(result1.binning_times[1]) / 1e6 << "ms"
            << std::endl;
  std::cout << "binning2 time: "
            << static_cast<double>(result1.binning_times[2]) / 1e6 << "ms"
            << std::endl;
  std::cout << "binning3 time: "
            << static_cast<double>(result1.binning_times[3]) / 1e6 << "ms"
            << std::endl;

  bool diff = false;
  for (int j = 0; j < result0.keys[3].size(); j++) {
    if (result0.keys[3][j] != result1.keys[3][j]) {
      diff = true;
      break;
    }
  }

  if (diff) {
    std::cout << std::endl;
    std::cout << "wrong" << std::endl;
    std::cout << std::endl;
    std::cout << "pass 3:" << std::endl;
    std::cout << std::hex;
    for (int j = 0; j < result1.keys[3].size() && j < 16; j++)
      std::cout << std::setfill('0') << std::setw(8) << result1.keys[3][j]
                << " ";
    std::cout << std::dec << std::endl;
  } else {
    std::cout << std::endl;
    std::cout << "ok" << std::endl;
    std::cout << std::endl;
  }

  return 0;
}
