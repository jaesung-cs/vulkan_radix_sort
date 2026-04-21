#ifndef VK_RADIX_SORT_FUCHSIA_BENCHMARK_H
#define VK_RADIX_SORT_FUCHSIA_BENCHMARK_H

#include "benchmark_base.h"

#include "volk.h"
#include "vk_mem_alloc.h"
#include <radix_sort/platforms/vk/radix_sort_vk.h>

class FuchsiaBenchmark : public BenchmarkBase {
 private:
  struct Buffer {
    VkBufferUsageFlags usage = 0;
    VkDeviceSize size = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    uint8_t* map = nullptr;
  };

 public:
  FuchsiaBenchmark();
  ~FuchsiaBenchmark() override;

  Results Sort(const std::vector<uint32_t>& keys) override;
  Results SortKeyValue(const std::vector<uint32_t>& keys,
                       const std::vector<uint32_t>& values) override;

 private:
  void Reallocate(Buffer* buffer, VkDeviceSize size, VkBufferUsageFlags usage, bool mapped = false);

  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  uint32_t queue_family_index_ = 0;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  float timestamp_period_ = 1.f;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  VkQueryPool query_pool_ = VK_NULL_HANDLE;

  radix_sort_vk_t* sorter_keys_ = nullptr;
  radix_sort_vk_t* sorter_kv_ = nullptr;

  Buffer staging_;
  Buffer keys_even_;
  Buffer keys_odd_;
  Buffer keys_internal_;
  Buffer kv_even_;
  Buffer kv_odd_;
  Buffer kv_internal_;
};

#endif  // VK_RADIX_SORT_FUCHSIA_BENCHMARK_H
