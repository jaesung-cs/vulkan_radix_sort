#include <iostream>
#include <iomanip>

#include "data_generator.h"
#include "cpu_benchmark_base.h"
#include "vulkan_benchmark_base.h"

int main() {
  std::cout << "vk_radix_sort benchmark" << std::endl;

  int size = 10000000;
  auto data = GenerateUniformRandomData(size);

  std::cout << "keys" << std::endl;
  std::cout << std::hex;
  for (int i = 0; i < size && i < 16; ++i) {
    std::cout << std::setfill('0') << std::setw(8) << data.keys[i] << " ";
  }
  std::cout << std::dec << std::endl;

  CpuBenchmarkBase cpu_benchmark;
  VulkanBenchmarkBase vulkan_benchmark;

  {
    std::cout << "================ sort ================" << std::endl;
    std::cout << "cpu sort" << std::endl;
    auto result0 = cpu_benchmark.Sort(data.keys);
    std::cout << "vulkan sort" << std::endl;
    auto result1 = vulkan_benchmark.Sort(data.keys);

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
    for (int j = 0; j < result0.keys[3].size(); ++j) {
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
      for (int j = 0; j < result1.keys[3].size() && j < 16; ++j)
        std::cout << std::setfill('0') << std::setw(8) << result1.keys[3][j]
                  << " ";
      std::cout << std::dec << std::endl;
    } else {
      std::cout << std::endl;
      std::cout << "ok" << std::endl;
      std::cout << std::endl;
    }
  }

  {
    std::cout << "================ sort key value ================"
              << std::endl;
    std::cout << "cpu sort" << std::endl;
    auto result0 = cpu_benchmark.SortKeyValue(data.keys, data.values);
    std::cout << "vulkan sort" << std::endl;
    auto result1 = vulkan_benchmark.SortKeyValue(data.keys, data.values);

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

    bool diff_keys = false;
    bool diff_values = false;
    int diff_value_location = 0;
    int diff_value_count = 0;
    for (int j = 0; j < result0.keys[3].size(); ++j) {
      if (result0.keys[3][j] != result1.keys[3][j]) {
        diff_keys = true;
      }
      if (result0.values[j] != result1.values[j]) {
        if (!diff_values) {
          diff_value_location = j;
        }
        diff_value_count++;
        diff_values = true;
      }
    }

    if (diff_keys || diff_values) {
      std::cout << std::endl;
      if (diff_keys) {
        std::cout << "wrong keys" << std::endl;
        std::cout << std::hex;
        for (int j = 0; j < result0.keys[3].size() && j < 16; ++j)
          std::cout << std::setfill('0') << std::setw(8) << result0.keys[3][j]
                    << " ";
        std::cout << std::endl;
        for (int j = 0; j < result1.keys[3].size() && j < 16; ++j)
          std::cout << std::setfill('0') << std::setw(8) << result1.keys[3][j]
                    << " ";
        std::cout << std::dec << std::endl;
      }
      if (diff_values) {
        std::cout << "wrong values" << std::endl;
        std::cout << "first location " << diff_value_location << std::endl;
        std::cout << "count " << diff_value_count << std::endl;
        std::cout << std::hex;
        for (int j = diff_value_location;
             j < size && j < diff_value_location + 16; ++j)
          std::cout << std::setfill('0') << std::setw(8) << result0.keys[3][j]
                    << " ";
        std::cout << std::endl;
        for (int j = diff_value_location;
             j < size && j < diff_value_location + 16; ++j)
          std::cout << std::setfill('0') << std::setw(8) << result0.values[j]
                    << " ";
        std::cout << std::endl;
        for (int j = diff_value_location;
             j < size && j < diff_value_location + 16; ++j)
          std::cout << std::setfill('0') << std::setw(8) << result1.keys[3][j]
                    << " ";
        std::cout << std::endl;
        for (int j = diff_value_location;
             j < size && j < diff_value_location + 16; ++j)
          std::cout << std::setfill('0') << std::setw(8) << result1.values[j]
                    << " ";
        std::cout << std::dec << std::endl;
      }
      std::cout << std::endl;
    } else {
      std::cout << std::endl;
      std::cout << "ok" << std::endl;
      std::cout << std::endl;
    }
  }

  {
    std::cout << "================ sort key value speed ================"
              << std::endl;

    for (int i = 0; i < 100; i++) {
      auto data = GenerateUniformRandomData(size);
      auto result1 = vulkan_benchmark.SortKeyValue(data.keys, data.values);

      std::cout << i << std::endl;
      std::cout << "total time: "
                << static_cast<double>(result1.total_time) / 1e6 << "ms"
                << std::endl;
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
    }
  }

  return 0;
}
