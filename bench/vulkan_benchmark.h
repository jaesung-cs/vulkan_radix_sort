#ifndef VK_RADIX_SORT_VULKAN_BENCHMARK_H
#define VK_RADIX_SORT_VULKAN_BENCHMARK_H

#include "benchmark_base.h"

#include <vulkan/vulkan.h>

#include "vk_mem_alloc.h"

class VulkanBenchmark : public BenchmarkBase {
 protected:
  struct Buffer {
    VkBufferUsageFlags usage = 0;
    VkDeviceSize size = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    uint8_t* map = nullptr;
  };

 public:
  VulkanBenchmark();
  ~VulkanBenchmark() override;

  BenchmarkResults Sort(const std::vector<uint32_t>& keys) override;
  BenchmarkResults SortKeyValue(const std::vector<uint32_t>& keys,
                                const std::vector<uint32_t>& values) override;

 protected:
  void Reallocate(Buffer* buffer, VkDeviceSize size, VkBufferUsageFlags usage,
                  bool mapped = false);

  virtual void SortGpu(VkCommandBuffer cb, size_t element_count) = 0;
  virtual void SortKeyValueGpu(VkCommandBuffer cb, size_t element_count) = 0;

 protected:
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
  VkQueryPool query_pool_ = VK_NULL_HANDLE;

  Buffer keys_;
  Buffer staging_;
};

#endif  // VK_RADIX_SORT_VULKAN_BENCHMARK_H
