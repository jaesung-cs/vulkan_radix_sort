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

  // histogram
  {
    std::cout << "================ global histogram ================"
              << std::endl;
    std::cout << "cpu global histogram" << std::endl;
    auto result0 = cpu_benchmark.GlobalHistogram(keys);
    std::cout << "vulkan global histogram" << std::endl;
    auto result1 = vulkan_benchmark.GlobalHistogram(keys);

    constexpr uint32_t RADIX = 256;
    for (int i = 0; i < 4; i++) {
      bool diff = false;
      for (int j = 0; j < RADIX; j++) {
        if (result0.histogram[j] != result1.histogram[j]) {
          diff = true;
          break;
        }
      }
      if (diff) {
        std::cout << "pass " << i << ":" << std::endl;
        for (int j = 0; j < RADIX; j++)
          std::cout << result0.histogram[i * RADIX + j] << ' ';
        std::cout << std::endl;
        for (int j = 0; j < RADIX; j++)
          std::cout << result1.histogram[i * RADIX + j] << ' ';
        std::cout << std::endl;
      }
    }
  }

  // sort steps
  {
    std::cout << "================ sort steps ================" << std::endl;
    std::cout << "vulkan sort steps" << std::endl;
    auto result1 = vulkan_benchmark.SortSteps(keys);

    for (int i = 0; i < 4; i++) {
      std::cout << "pass " << i << ":" << std::endl;
      std::cout << std::hex;
      for (int j = 0; j < result1.keys[i].size() && j < 16; j++)
        std::cout << std::setfill('0') << std::setw(8) << result1.keys[i][j]
                  << " ";
      std::cout << std::dec << std::endl;
    }

    bool wrong = false;
    for (int j = 1; j < result1.keys[3].size(); j++) {
      if (result1.keys[3][j - 1] > result1.keys[3][j]) {
        wrong = true;
        break;
      }
    }

    if (wrong)
      std::cout << "wrong" << std::endl;
    else
      std::cout << "ok" << std::endl;
  }

  // sort
  {
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
      std::cout << "wrong" << std::endl;
      std::cout << "pass 3:" << std::endl;
      std::cout << std::hex;
      for (int j = 0; j < result1.keys[3].size() && j < 16; j++)
        std::cout << std::setfill('0') << std::setw(8) << result1.keys[3][j]
                  << " ";
      std::cout << std::dec << std::endl;
    } else {
      std::cout << "ok" << std::endl;
    }
  }

  return 0;
}
