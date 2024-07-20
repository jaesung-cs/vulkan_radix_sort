#ifndef VK_RADIX_SORT_VULKAN_BENCHMARK_H
#define VK_RADIX_SORT_VULKAN_BENCHMARK_H

#include "benchmark_base.h"

#include <vulkan/vulkan.h>

#include "vk_mem_alloc.h"

#include <vk_radix_sort.h>

class VulkanBenchmark : public BenchmarkBase {
 public:
  VulkanBenchmark();
  ~VulkanBenchmark() override;

  Results Sort(const std::vector<uint32_t>& keys) override;
  Results SortKeyValue(const std::vector<uint32_t>& keys,
                       const std::vector<uint32_t>& values) override;

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
  VrdxSorter sorter_ = VK_NULL_HANDLE;
  VkQueryPool query_pool_ = VK_NULL_HANDLE;

  struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    uint8_t* map = nullptr;
  };
  Buffer keys_;
  Buffer storage_;
  Buffer staging_;
};

#endif  // VK_RADIX_SORT_VULKAN_BENCHMARK_H
