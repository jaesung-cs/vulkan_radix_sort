#ifndef VK_RADIX_SORT_VULKAN_BENCHMARK_BASE_H
#define VK_RADIX_SORT_VULKAN_BENCHMARK_BASE_H

#include <vulkan/vulkan.h>

#include "vk_mem_alloc.h"

#include <vk_radix_sort.h>

class VulkanBenchmarkBase {
 public:
  VulkanBenchmarkBase();
  ~VulkanBenchmarkBase();

 private:
  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  uint32_t queue_family_index_ = 0;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  VxSorter sorter_ = VK_NULL_HANDLE;
};

#endif  // VK_RADIX_SORT_VULKAN_BENCHMARK_BASE_H
