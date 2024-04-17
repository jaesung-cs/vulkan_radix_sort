#ifndef VK_RADIX_SORT_CPU_BENCHMARK_BASE_H
#define VK_RADIX_SORT_CPU_BENCHMARK_BASE_H

#include <vector>

#include <vulkan/vulkan.h>

#include "vk_mem_alloc.h"

#include <vk_radix_sort.h>

class CpuBenchmarkBase {
 private:
  struct IntermediateResults {
    std::vector<uint32_t> histogram;
    std::vector<uint32_t> keys[4];
  };

 public:
  CpuBenchmarkBase();
  ~CpuBenchmarkBase();

  IntermediateResults GlobalHistogram(const std::vector<uint32_t>& keys);
  IntermediateResults Sort(const std::vector<uint32_t>& keys);

 private:
};

#endif  // VK_RADIX_SORT_CPU_BENCHMARK_BASE_H
