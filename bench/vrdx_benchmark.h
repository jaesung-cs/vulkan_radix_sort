#ifndef VK_RADIX_SORT_VRDX_BENCHMARK_H
#define VK_RADIX_SORT_VRDX_BENCHMARK_H

#include "vulkan_benchmark.h"

#include <vk_radix_sort.h>

class VrdxBenchmark : public VulkanBenchmark {
 public:
  VrdxBenchmark();
  ~VrdxBenchmark();

 protected:
  void SortGpu(VkCommandBuffer cb, size_t element_count) override;
  void SortKeyValueGpu(VkCommandBuffer cb, size_t element_count) override;

 private:
  VrdxSorter sorter_ = VK_NULL_HANDLE;
  Buffer storage_;
};

#endif  // VK_RADIX_SORT_VRDX_BENCHMARK_H
