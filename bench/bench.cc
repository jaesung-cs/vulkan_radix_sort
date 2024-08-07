#include <iostream>
#include <iomanip>

#include "data_generator.h"
#include "cuda_benchmark.h"
#include "vulkan_benchmark.h"

int main() {
  std::cout << "vk_radix_sort benchmark" << std::endl;

  // target: 15 GItems/s for key, 11 GItems/s for kv sort, 4.19e6 items (A100)
  int size = 33554432;

  CudaBenchmark cuda_benchmark;
  VulkanBenchmark vulkan_benchmark;

  {
    std::cout << "================ sort ================" << std::endl;
    auto data = GenerateUniformRandomData(size);

    std::cout << "cuda sort" << std::endl;
    auto result0 = cuda_benchmark.Sort(data.keys);
    {
      double perf = (static_cast<double>(size) / 1e9) /
                    (static_cast<double>(result0.total_time) / 1e9);
      std::cout << "total time: "
                << static_cast<double>(result0.total_time) / 1e6 << "ms ("
                << perf << " GItems/s)" << std::endl;
    }

    std::cout << "vulkan sort" << std::endl;
    auto result1 = vulkan_benchmark.Sort(data.keys);
    {
      double perf = (static_cast<double>(size) / 1e9) /
                    (static_cast<double>(result1.total_time) / 1e9);

      std::cout << "total time: "
                << static_cast<double>(result1.total_time) / 1e6 << "ms ("
                << perf << " GItems/s)" << std::endl;
    }

    bool diff = false;
    int diff_location = -1;
    for (int j = 0; j < result0.keys.size(); ++j) {
      if (result0.keys[j] != result1.keys[j]) {
        diff = true;
        if (diff_location == -1) {
          diff_location = j;
        }
      }
    }

    if (diff) {
      std::cout << std::endl;
      std::cout << "wrong" << std::endl;
      std::cout << std::endl;
      std::cout << "first location " << diff_location << std::endl;
      std::cout << "pass 3:" << std::endl;
      std::cout << std::hex;
      for (int j = diff_location;
           j < result0.keys.size() && j < diff_location + 16; ++j)
        std::cout << std::setfill('0') << std::setw(8) << result0.keys[j]
                  << " ";
      std::cout << std::endl;
      for (int j = diff_location;
           j < result1.keys.size() && j < diff_location + 16; ++j)
        std::cout << std::setfill('0') << std::setw(8) << result1.keys[j]
                  << " ";
      std::cout << std::dec << std::endl;

      std::cout << "pass 3:" << std::endl;
      std::cout << std::hex;
      for (int j = std::max(size - 16, 0); j < size; ++j)
        std::cout << std::setfill('0') << std::setw(8) << result0.keys[j]
                  << " ";
      std::cout << std::endl;
      for (int j = std::max(size - 16, 0); j < size; ++j)
        std::cout << std::setfill('0') << std::setw(8) << result1.keys[j]
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
    auto data = GenerateUniformRandomData(size);
    std::cout << "cuda sort" << std::endl;
    auto result0 = cuda_benchmark.SortKeyValue(data.keys, data.values);
    {
      double perf = (static_cast<double>(size) / 1e9) /
                    (static_cast<double>(result0.total_time) / 1e9);
      std::cout << "total time: "
                << static_cast<double>(result0.total_time) / 1e6 << "ms ("
                << perf << " GItems/s)" << std::endl;
    }

    std::cout << "vulkan sort" << std::endl;
    auto result1 = vulkan_benchmark.SortKeyValue(data.keys, data.values);
    {
      double perf = (static_cast<double>(size) / 1e9) /
                    (static_cast<double>(result1.total_time) / 1e9);

      std::cout << "total time: "
                << static_cast<double>(result1.total_time) / 1e6 << "ms ("
                << perf << " GItems/s)" << std::endl;
    }

    bool diff_keys = false;
    bool diff_values = false;
    int diff_value_location = 0;
    int diff_value_count = 0;
    for (int j = 0; j < result0.keys.size(); ++j) {
      if (result0.keys[j] != result1.keys[j]) {
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
        for (int j = 0; j < result0.keys.size() && j < 16; ++j)
          std::cout << std::setfill('0') << std::setw(8) << result0.keys[j]
                    << " ";
        std::cout << std::endl;
        for (int j = 0; j < result1.keys.size() && j < 16; ++j)
          std::cout << std::setfill('0') << std::setw(8) << result1.keys[j]
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
          std::cout << std::setfill('0') << std::setw(8) << result0.keys[j]
                    << " ";
        std::cout << std::endl;
        for (int j = diff_value_location;
             j < size && j < diff_value_location + 16; ++j)
          std::cout << std::setfill('0') << std::setw(8) << result0.values[j]
                    << " ";
        std::cout << std::endl;
        for (int j = diff_value_location;
             j < size && j < diff_value_location + 16; ++j)
          std::cout << std::setfill('0') << std::setw(8) << result1.keys[j]
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

    for (int i = 0; i < 100; ++i) {
      auto data = GenerateUniformRandomData(size);
      auto result0 = cuda_benchmark.SortKeyValue(data.keys, data.values);
      auto result1 = vulkan_benchmark.SortKeyValue(data.keys, data.values);

      double perf0 = (static_cast<double>(size) / 1e9) /
                     (static_cast<double>(result0.total_time) / 1e9);
      double perf1 = (static_cast<double>(size) / 1e9) /
                     (static_cast<double>(result1.total_time) / 1e9);

      std::cout << "[" << i << "] total time: CUB "
                << static_cast<double>(result0.total_time) / 1e6 << "ms ("
                << perf0 << " GItems/s) vs. Vulkan "
                << static_cast<double>(result1.total_time) / 1e6 << "ms ("
                << perf1 << " GItems/s)" << std::endl;
    }
  }

  return 0;
}
