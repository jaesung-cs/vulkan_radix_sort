#ifndef VK_RADIX_SORT_VULKAN_BENCHMARK_BASE_H
#define VK_RADIX_SORT_VULKAN_BENCHMARK_BASE_H

#include <vector>

#include <vulkan/vulkan.h>

#include "vk_mem_alloc.h"

#include <vk_radix_sort.h>

class VulkanBenchmarkBase {
 private:
  struct IntermediateResults {
    std::vector<uint32_t> keys[4];
    std::vector<uint32_t> values;

    uint64_t total_time = 0;
    std::vector<uint64_t> reduce_then_scan_times;
  };

 public:
  VulkanBenchmarkBase();
  ~VulkanBenchmarkBase();

  IntermediateResults Sort(const std::vector<uint32_t>& keys);
  IntermediateResults SortKeyValue(const std::vector<uint32_t>& keys,
                                   const std::vector<uint32_t>& values);

 private:
  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  uint32_t queue_family_index_ = 0;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  VrdxSorterLayout sorter_layout_ = VK_NULL_HANDLE;
  VrdxSorter sorter_ = VK_NULL_HANDLE;
  VkQueryPool query_pool_ = VK_NULL_HANDLE;

  static constexpr uint32_t MAX_ELEMENT_COUNT = 1 << 24;
  static constexpr uint32_t PARTITION_SIZE = 7680;

  struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    uint8_t* map = nullptr;
  };
  Buffer keys_;
  Buffer indirect_;
  Buffer staging_;
};

#endif  // VK_RADIX_SORT_VULKAN_BENCHMARK_BASE_H
