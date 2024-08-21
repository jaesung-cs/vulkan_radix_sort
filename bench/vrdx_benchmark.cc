#include "vrdx_benchmark.h"

#include <iostream>

VrdxBenchmark::VrdxBenchmark() : VulkanBenchmark() {
  // sorter
  VrdxSorterCreateInfo sorter_info = {};
  sorter_info.physicalDevice = physical_device_;
  sorter_info.device = device_;
  vrdxCreateSorter(&sorter_info, &sorter_);
}

VrdxBenchmark::~VrdxBenchmark() {
  if (storage_.buffer)
    vmaDestroyBuffer(allocator_, storage_.buffer, storage_.allocation);
  vrdxDestroySorter(sorter_);
}

void VrdxBenchmark::SortGpu(VkCommandBuffer cb, size_t element_count) {
  VrdxSorterStorageRequirements requirements;
  vrdxGetSorterStorageRequirements(sorter_, element_count, &requirements);
  Reallocate(&storage_, requirements.size, requirements.usage);

  vrdxCmdSort(cb, sorter_, element_count, keys_.buffer, 0, storage_.buffer, 0,
              query_pool_, 0);
}

void VrdxBenchmark::SortKeyValueGpu(VkCommandBuffer cb, size_t element_count) {
  VrdxSorterStorageRequirements requirements;
  vrdxGetSorterKeyValueStorageRequirements(sorter_, element_count,
                                           &requirements);
  Reallocate(&storage_, requirements.size, requirements.usage);

  vrdxCmdSortKeyValueIndirect(
      cb, sorter_, element_count, keys_.buffer,
      2 * element_count * sizeof(uint32_t), keys_.buffer, 0, keys_.buffer,
      element_count * sizeof(uint32_t), storage_.buffer, 0, query_pool_, 0);
}
