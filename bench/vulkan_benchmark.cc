#include "vulkan_benchmark.h"

#include <iostream>
#include <cstring>
#include <chrono>

namespace {

uint32_t Align(uint32_t a, uint32_t b) { return (a + b - 1) / b * b; }

constexpr auto timestamp_count = 15;
constexpr auto timing_query_count = 2;

static VKAPI_ATTR VkBool32 VKAPI_CALL
DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl << std::endl;

  return VK_FALSE;
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
                                      const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator,
                                      VkDebugUtilsMessengerEXT* pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks* pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

}  // namespace

std::string VulkanBenchmark::LibraryVersion() const {
  return "v" + std::to_string(VRDX_VERSION_MAJOR) + "." + std::to_string(VRDX_VERSION_MINOR) + "." +
         std::to_string(VRDX_VERSION_PATCH);
}

VulkanBenchmark::VulkanBenchmark(bool validation, bool timestamps) {
  volkInitialize();

  // instance
  VkApplicationInfo application_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vk_radix_sort_benchmark",
      .applicationVersion =
          VK_MAKE_API_VERSION(VRDX_VERSION_MAJOR, VRDX_VERSION_MINOR, VRDX_VERSION_PATCH, 0),
      .pEngineName = "vk_radix_sort",
      .engineVersion =
          VK_MAKE_API_VERSION(VRDX_VERSION_MAJOR, VRDX_VERSION_MINOR, VRDX_VERSION_PATCH, 0),
      .apiVersion = VK_API_VERSION_1_4,
  };

  VkDebugUtilsMessengerCreateInfoEXT messenger_info = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
      .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
      .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
      .pfnUserCallback = DebugCallback,
  };

  std::vector<const char*> layers;
  std::vector<const char*> instance_extensions = {
#ifdef __APPLE__
      VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
#endif
  };

  if (validation) {
    layers.push_back("VK_LAYER_KHRONOS_validation");
    instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pNext = validation ? &messenger_info : nullptr,
#ifdef __APPLE__
      .flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
#endif
      .pApplicationInfo = &application_info,
      .enabledLayerCount = static_cast<uint32_t>(layers.size()),
      .ppEnabledLayerNames = layers.data(),
      .enabledExtensionCount = static_cast<uint32_t>(instance_extensions.size()),
      .ppEnabledExtensionNames = instance_extensions.data(),
  };
  vkCreateInstance(&instance_info, NULL, &instance_);
  volkLoadInstance(instance_);

  if (validation) CreateDebugUtilsMessengerEXT(instance_, &messenger_info, NULL, &messenger_);

  // physical device
  uint32_t physical_device_count = 0;
  vkEnumeratePhysicalDevices(instance_, &physical_device_count, NULL);
  std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
  vkEnumeratePhysicalDevices(instance_, &physical_device_count, physical_devices.data());
  physical_device_ = physical_devices[0];

  VkPhysicalDeviceProperties device_properties;
  vkGetPhysicalDeviceProperties(physical_device_, &device_properties);
  min_buffer_alignment_ =
      static_cast<uint32_t>(device_properties.limits.minStorageBufferOffsetAlignment);
  timestamp_period_ = device_properties.limits.timestampPeriod;

  // find graphics queue
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, NULL);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count,
                                           queue_families.data());

  for (size_t i = 0; i < queue_families.size(); ++i) {
    const auto& queue_family = queue_families[i];
    if ((queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT) == VK_QUEUE_COMPUTE_BIT &&
        (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
      queue_family_index_ = i;
      break;
    }
  }

  // queues
  std::vector<float> queue_priorities = {
      1.f,
  };
  std::vector<VkDeviceQueueCreateInfo> queue_infos(1);
  queue_infos[0] = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family_index_,
      .queueCount = static_cast<uint32_t>(queue_priorities.size()),
      .pQueuePriorities = queue_priorities.data(),
  };

  std::vector<const char*> device_extensions = {
#ifdef __APPLE__
      "VK_KHR_portability_subset",
#endif
  };

  VkPhysicalDeviceVulkan13Features features13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .synchronization2 = VK_TRUE,
  };

  VkPhysicalDeviceVulkan14Features features14 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
      .pNext = &features13,
      .pushDescriptor = VK_TRUE,
  };

  VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &features14,
      .queueCreateInfoCount = static_cast<uint32_t>(queue_infos.size()),
      .pQueueCreateInfos = queue_infos.data(),
      .enabledExtensionCount = static_cast<uint32_t>(device_extensions.size()),
      .ppEnabledExtensionNames = device_extensions.data(),
  };
  vkCreateDevice(physical_device_, &device_info, NULL, &device_);
  volkLoadDevice(device_);

  vkGetDeviceQueue(device_, queue_family_index_, 0, &queue_);

  // vma
  VmaVulkanFunctions functions = {
      .vkGetInstanceProcAddr = vkGetInstanceProcAddr,
      .vkGetDeviceProcAddr = vkGetDeviceProcAddr,
  };

  VmaAllocatorCreateInfo allocator_info = {
      .physicalDevice = physical_device_,
      .device = device_,
      .pVulkanFunctions = &functions,
      .instance = instance_,
      .vulkanApiVersion = application_info.apiVersion,
  };
  vmaCreateAllocator(&allocator_info, &allocator_);

  // commands
  VkCommandPoolCreateInfo command_pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags =
          VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = queue_family_index_,
  };
  vkCreateCommandPool(device_, &command_pool_info, NULL, &command_pool_);

  VkCommandBufferAllocateInfo command_buffer_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool_,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  vkAllocateCommandBuffers(device_, &command_buffer_info, &command_buffer_);

  // fence
  VkFenceCreateInfo fence_info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  vkCreateFence(device_, &fence_info, NULL, &fence_);

  // timestamp query pool (optional; adds vkCmdWriteTimestamp overhead to each sort)
  if (timestamps) {
    VkQueryPoolCreateInfo query_pool_info = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = timestamp_count,
    };
    vkCreateQueryPool(device_, &query_pool_info, NULL, &query_pool_);
  }

  // lightweight always-on query pool for total GPU time, matching cuda/fuchsia benchmarks
  VkQueryPoolCreateInfo timing_pool_info = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = timing_query_count,
  };
  vkCreateQueryPool(device_, &timing_pool_info, NULL, &timing_pool_);

  // sorter
  VrdxSorterCreateInfo sorter_info = {
      .physicalDevice = physical_device_,
      .device = device_,
  };
  vrdxCreateSorter(&sorter_info, &sorter_);
}

VulkanBenchmark::~VulkanBenchmark() {
  vkDeviceWaitIdle(device_);

  if (keys_.buffer) vmaDestroyBuffer(allocator_, keys_.buffer, keys_.allocation);
  if (values_.buffer) vmaDestroyBuffer(allocator_, values_.buffer, values_.allocation);
  if (storage_.buffer) vmaDestroyBuffer(allocator_, storage_.buffer, storage_.allocation);
  if (staging_.buffer) vmaDestroyBuffer(allocator_, staging_.buffer, staging_.allocation);

  vrdxDestroySorter(sorter_);
  vkDestroyQueryPool(device_, query_pool_, NULL);
  vkDestroyQueryPool(device_, timing_pool_, NULL);
  vkDestroyFence(device_, fence_, NULL);
  vkDestroyCommandPool(device_, command_pool_, NULL);
  vmaDestroyAllocator(allocator_);
  vkDestroyDevice(device_, NULL);
  DestroyDebugUtilsMessengerEXT(instance_, messenger_, NULL);
  vkDestroyInstance(instance_, NULL);

  volkFinalize();
}

void VulkanBenchmark::Reallocate(Buffer* buffer, VkDeviceSize size, VkBufferUsageFlags usage,
                                 bool mapped) {
  if ((buffer->usage & usage) == usage && buffer->size >= size &&
      (mapped && buffer->map || !mapped && buffer->map == nullptr))
    return;

  if (buffer->allocation) vmaDestroyBuffer(allocator_, buffer->buffer, buffer->allocation);

  VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
  };
  VmaAllocationCreateInfo allocation_create_info = {
      .usage = VMA_MEMORY_USAGE_AUTO,
  };
  if (mapped) {
    allocation_create_info.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
  }

  VmaAllocationInfo allocation_info;
  vmaCreateBuffer(allocator_, &buffer_info, &allocation_create_info, &buffer->buffer,
                  &buffer->allocation, &allocation_info);

  buffer->usage = usage;
  buffer->size = size;
  if (mapped) buffer->map = reinterpret_cast<uint8_t*>(allocation_info.pMappedData);
}

VulkanBenchmark::Results VulkanBenchmark::Sort(const std::vector<uint32_t>& keys) {
  uint32_t element_count = keys.size();
  uint32_t inout_size = Align(element_count * sizeof(uint32_t), min_buffer_alignment_);

  Reallocate(&staging_, inout_size,
             VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  Reallocate(&keys_, inout_size,
             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  VrdxSorterStorageRequirements requirements;
  vrdxGetSorterStorageRequirements(sorter_, element_count, VRDX_SORT_MODE_KEYS_ONLY, &requirements);
  Reallocate(&storage_, requirements.size, requirements.usage);

  std::memcpy(staging_.map, keys.data(), element_count * sizeof(uint32_t));

  VkCommandBufferBeginInfo command_buffer_begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  vkBeginCommandBuffer(command_buffer_, &command_buffer_begin_info);

  if (query_pool_) vkCmdResetQueryPool(command_buffer_, query_pool_, 0, timestamp_count);

  // copy to keys buffer
  VkBufferCopy region = {
      .srcOffset = 0,
      .dstOffset = 0,
      .size = element_count * sizeof(uint32_t),
  };
  vkCmdCopyBuffer(command_buffer_, staging_.buffer, keys_.buffer, 1, &region);

  vkEndCommandBuffer(command_buffer_);

  VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer_,
  };
  vkQueueSubmit(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  vkResetFences(device_, 1, &fence_);

  // sort
  vkBeginCommandBuffer(command_buffer_, &command_buffer_begin_info);

  vkCmdResetQueryPool(command_buffer_, timing_pool_, 0, timing_query_count);
  vkCmdWriteTimestamp(command_buffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timing_pool_, 0);

  VrdxSortInfo sort_info = {
      .elementCount = element_count,
      .keysBuffer = keys_.buffer,
      .storageBuffer = storage_.buffer,
      .queryPool = query_pool_,
  };
  vrdxCmdSort(command_buffer_, sorter_, &sort_info);

  vkCmdWriteTimestamp(command_buffer_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool_, 1);

  vkEndCommandBuffer(command_buffer_);
  auto cpu_start = std::chrono::steady_clock::now();
  vkQueueSubmit(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  auto cpu_end = std::chrono::steady_clock::now();
  vkResetFences(device_, 1, &fence_);

  // copy back
  vkBeginCommandBuffer(command_buffer_, &command_buffer_begin_info);

  region = {
      .srcOffset = 0,
      .dstOffset = 0,
      .size = element_count * sizeof(uint32_t),
  };
  vkCmdCopyBuffer(command_buffer_, keys_.buffer, staging_.buffer, 1, &region);

  vkEndCommandBuffer(command_buffer_);
  vkQueueSubmit(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  vkResetFences(device_, 1, &fence_);

  Results result;
  result.keys.resize(element_count);
  std::memcpy(result.keys.data(), staging_.map, element_count * sizeof(uint32_t));
  result.cpu_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count();

  auto ticks_to_ns = [&](uint64_t ticks) -> uint64_t {
    return static_cast<uint64_t>(ticks * timestamp_period_);
  };

  uint64_t timing[timing_query_count];
  vkGetQueryPoolResults(device_, timing_pool_, 0, timing_query_count, sizeof(timing), timing,
                        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
  result.total_time = ticks_to_ns(timing[1] - timing[0]);

  if (query_pool_) {
    std::vector<uint64_t> timestamps(timestamp_count);
    vkGetQueryPoolResults(device_, query_pool_, 0, timestamps.size(),
                          timestamps.size() * sizeof(uint64_t), timestamps.data(), sizeof(uint64_t),
                          VK_QUERY_RESULT_64_BIT);

    for (int pass = 0; pass < 4; ++pass) {
      result.upsweep_ns += ticks_to_ns(timestamps[2 + 3 * pass] - timestamps[1 + 3 * pass]);
      result.spine_ns += ticks_to_ns(timestamps[3 + 3 * pass] - timestamps[2 + 3 * pass]);
      result.downsweep_ns += ticks_to_ns(timestamps[4 + 3 * pass] - timestamps[3 + 3 * pass]);
    }
  }
  return result;
}

VulkanBenchmark::Results VulkanBenchmark::SortKeyValue(const std::vector<uint32_t>& keys,
                                                       const std::vector<uint32_t>& values) {
  uint32_t element_count = keys.size();
  uint32_t inout_size = Align(element_count * sizeof(uint32_t), min_buffer_alignment_);

  Reallocate(&staging_, 2 * inout_size,
             VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  Reallocate(&keys_, inout_size,
             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  Reallocate(&values_, inout_size,
             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  VrdxSorterStorageRequirements requirements;
  vrdxGetSorterStorageRequirements(sorter_, element_count, VRDX_SORT_MODE_KEY_VALUE, &requirements);
  Reallocate(&storage_, requirements.size, requirements.usage);

  std::memcpy(staging_.map, keys.data(), element_count * sizeof(uint32_t));
  std::memcpy(staging_.map + inout_size, values.data(), element_count * sizeof(uint32_t));

  VkCommandBufferBeginInfo command_buffer_begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  vkBeginCommandBuffer(command_buffer_, &command_buffer_begin_info);

  if (query_pool_) vkCmdResetQueryPool(command_buffer_, query_pool_, 0, timestamp_count);

  // copy to keys/values buffers
  VkBufferCopy keys_region = {
      .srcOffset = 0,
      .dstOffset = 0,
      .size = inout_size,
  };
  vkCmdCopyBuffer(command_buffer_, staging_.buffer, keys_.buffer, 1, &keys_region);

  VkBufferCopy values_region = {
      .srcOffset = inout_size,
      .dstOffset = 0,
      .size = inout_size,
  };
  vkCmdCopyBuffer(command_buffer_, staging_.buffer, values_.buffer, 1, &values_region);

  vkEndCommandBuffer(command_buffer_);

  VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer_,
  };
  vkQueueSubmit(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  vkResetFences(device_, 1, &fence_);

  // sort
  vkBeginCommandBuffer(command_buffer_, &command_buffer_begin_info);

  vkCmdResetQueryPool(command_buffer_, timing_pool_, 0, timing_query_count);
  vkCmdWriteTimestamp(command_buffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timing_pool_, 0);

  VrdxSortInfo sort_info = {
      .elementCount = element_count,
      .keysBuffer = keys_.buffer,
      .valuesBuffer = values_.buffer,
      .storageBuffer = storage_.buffer,
      .queryPool = query_pool_,
  };
  vrdxCmdSort(command_buffer_, sorter_, &sort_info);

  vkCmdWriteTimestamp(command_buffer_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool_, 1);

  vkEndCommandBuffer(command_buffer_);
  auto cpu_start = std::chrono::steady_clock::now();
  vkQueueSubmit(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  auto cpu_end = std::chrono::steady_clock::now();
  vkResetFences(device_, 1, &fence_);

  // copy back
  vkBeginCommandBuffer(command_buffer_, &command_buffer_begin_info);

  VkBufferCopy keys_back_region = {
      .srcOffset = 0,
      .dstOffset = 0,
      .size = inout_size,
  };
  vkCmdCopyBuffer(command_buffer_, keys_.buffer, staging_.buffer, 1, &keys_back_region);

  VkBufferCopy values_back_region = {
      .srcOffset = 0,
      .dstOffset = inout_size,
      .size = inout_size,
  };
  vkCmdCopyBuffer(command_buffer_, values_.buffer, staging_.buffer, 1, &values_back_region);

  vkEndCommandBuffer(command_buffer_);
  vkQueueSubmit(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  vkResetFences(device_, 1, &fence_);

  Results result;
  result.keys.resize(element_count);
  result.values.resize(element_count);
  std::memcpy(result.keys.data(), staging_.map, element_count * sizeof(uint32_t));
  std::memcpy(result.values.data(), staging_.map + inout_size, element_count * sizeof(uint32_t));
  result.cpu_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count();

  auto ticks_to_ns = [&](uint64_t ticks) -> uint64_t {
    return static_cast<uint64_t>(ticks * timestamp_period_);
  };

  uint64_t timing[timing_query_count];
  vkGetQueryPoolResults(device_, timing_pool_, 0, timing_query_count, sizeof(timing), timing,
                        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
  result.total_time = ticks_to_ns(timing[1] - timing[0]);

  if (query_pool_) {
    std::vector<uint64_t> timestamps(timestamp_count);
    vkGetQueryPoolResults(device_, query_pool_, 0, timestamps.size(),
                          timestamps.size() * sizeof(uint64_t), timestamps.data(), sizeof(uint64_t),
                          VK_QUERY_RESULT_64_BIT);

    for (int pass = 0; pass < 4; ++pass) {
      result.upsweep_ns += ticks_to_ns(timestamps[2 + 3 * pass] - timestamps[1 + 3 * pass]);
      result.spine_ns += ticks_to_ns(timestamps[3 + 3 * pass] - timestamps[2 + 3 * pass]);
      result.downsweep_ns += ticks_to_ns(timestamps[4 + 3 * pass] - timestamps[3 + 3 * pass]);
    }
  }
  return result;
}
