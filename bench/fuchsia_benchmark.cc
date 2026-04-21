#include "fuchsia_benchmark.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

constexpr uint32_t kTimestampCount = 2;

static VKAPI_ATTR VkBool32 VKAPI_CALL
DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
              const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl << std::endl;
  return VK_FALSE;
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
                                      const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator,
                                      VkDebugUtilsMessengerEXT* pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT messenger,
                                   const VkAllocationCallbacks* pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) func(instance, messenger, pAllocator);
}

}  // namespace

FuchsiaBenchmark::FuchsiaBenchmark() {
  volkInitialize();

  // instance
  VkApplicationInfo app_info = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app_info.pApplicationName = "vk_radix_sort_benchmark";
  app_info.apiVersion = VK_API_VERSION_1_2;

  VkDebugUtilsMessengerCreateInfoEXT messenger_info = {
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
  messenger_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                   VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  messenger_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  messenger_info.pfnUserCallback = DebugCallback;

  std::vector<const char*> layers = {"VK_LAYER_KHRONOS_validation"};
  std::vector<const char*> instance_extensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};

  VkInstanceCreateInfo instance_info = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instance_info.pNext = &messenger_info;
  instance_info.pApplicationInfo = &app_info;
  instance_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
  instance_info.ppEnabledLayerNames = layers.data();
  instance_info.enabledExtensionCount = static_cast<uint32_t>(instance_extensions.size());
  instance_info.ppEnabledExtensionNames = instance_extensions.data();
  vkCreateInstance(&instance_info, nullptr, &instance_);
  volkLoadInstance(instance_);

  CreateDebugUtilsMessengerEXT(instance_, &messenger_info, nullptr, &messenger_);

  // physical device
  uint32_t physical_device_count = 0;
  vkEnumeratePhysicalDevices(instance_, &physical_device_count, nullptr);
  std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
  vkEnumeratePhysicalDevices(instance_, &physical_device_count, physical_devices.data());
  physical_device_ = physical_devices[0];

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(physical_device_, &props);
  timestamp_period_ = props.limits.timestampPeriod;

  // find compute queue
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count,
                                           queue_families.data());
  for (size_t i = 0; i < queue_families.size(); ++i) {
    if ((queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
        !(queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
      queue_family_index_ = static_cast<uint32_t>(i);
      break;
    }
  }

  // auto-detect fuchsia targets
  auto target_keys = radix_sort_vk_target_auto_detect(&props, 1);
  auto target_kv = radix_sort_vk_target_auto_detect(&props, 2);

  // query device requirements from the 2-dword target (superset)
  VkPhysicalDeviceFeatures pdf = {};
  VkPhysicalDeviceVulkan11Features pdf11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceVulkan12Features pdf12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};

  radix_sort_vk_target_requirements_t reqs = {};
  reqs.pdf = &pdf;
  reqs.pdf11 = &pdf11;
  reqs.pdf12 = &pdf12;
  radix_sort_vk_target_get_requirements(target_kv, &reqs);  // first call: get count

  std::vector<const char*> device_extensions(reqs.ext_name_count);
  reqs.ext_names = device_extensions.data();
  radix_sort_vk_target_get_requirements(target_kv, &reqs);  // second call: fill names

  // chain features into device create info
  VkPhysicalDeviceFeatures2 features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features2.features = pdf;
  features2.pNext = &pdf11;
  pdf11.pNext = &pdf12;

  float queue_priority = 1.f;
  VkDeviceQueueCreateInfo queue_info = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue_info.queueFamilyIndex = queue_family_index_;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &queue_priority;

  VkDeviceCreateInfo device_info = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device_info.pNext = &features2;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  device_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
  device_info.ppEnabledExtensionNames = device_extensions.data();
  vkCreateDevice(physical_device_, &device_info, nullptr, &device_);
  volkLoadDevice(device_);

  vkGetDeviceQueue(device_, queue_family_index_, 0, &queue_);

  // vma
  VmaVulkanFunctions vma_funcs = {};
  vma_funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  vma_funcs.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

  VmaAllocatorCreateInfo allocator_info = {};
  allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  allocator_info.physicalDevice = physical_device_;
  allocator_info.device = device_;
  allocator_info.instance = instance_;
  allocator_info.pVulkanFunctions = &vma_funcs;
  allocator_info.vulkanApiVersion = VK_API_VERSION_1_2;
  vmaCreateAllocator(&allocator_info, &allocator_);

  // command pool / buffer
  VkCommandPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.flags =
      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  pool_info.queueFamilyIndex = queue_family_index_;
  vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_);

  VkCommandBufferAllocateInfo cmd_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cmd_info.commandPool = command_pool_;
  cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_info.commandBufferCount = 1;
  vkAllocateCommandBuffers(device_, &cmd_info, &command_buffer_);

  VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  vkCreateFence(device_, &fence_info, nullptr, &fence_);

  VkQueryPoolCreateInfo qp_info = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  qp_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qp_info.queryCount = kTimestampCount;
  vkCreateQueryPool(device_, &qp_info, nullptr, &query_pool_);

  // sorters
  sorter_keys_ = radix_sort_vk_create(device_, nullptr, nullptr, target_keys);
  sorter_kv_ = radix_sort_vk_create(device_, nullptr, nullptr, target_kv);
}

FuchsiaBenchmark::~FuchsiaBenchmark() {
  vkDeviceWaitIdle(device_);

  auto destroy = [&](Buffer& b) {
    if (b.buffer) vmaDestroyBuffer(allocator_, b.buffer, b.allocation);
  };
  destroy(staging_);
  destroy(keys_even_);
  destroy(keys_odd_);
  destroy(keys_internal_);
  destroy(kv_even_);
  destroy(kv_odd_);
  destroy(kv_internal_);

  radix_sort_vk_destroy(sorter_keys_, device_, nullptr);
  radix_sort_vk_destroy(sorter_kv_, device_, nullptr);
  vkDestroyQueryPool(device_, query_pool_, nullptr);
  vkDestroyFence(device_, fence_, nullptr);
  vkDestroyCommandPool(device_, command_pool_, nullptr);
  vmaDestroyAllocator(allocator_);
  vkDestroyDevice(device_, nullptr);
  DestroyDebugUtilsMessengerEXT(instance_, messenger_, nullptr);
  vkDestroyInstance(instance_, nullptr);
  volkFinalize();
}

void FuchsiaBenchmark::Reallocate(Buffer* buffer, VkDeviceSize size, VkBufferUsageFlags usage,
                                  bool mapped) {
  if ((buffer->usage & usage) == usage && buffer->size >= size &&
      (mapped ? buffer->map != nullptr : buffer->map == nullptr))
    return;

  if (buffer->allocation) vmaDestroyBuffer(allocator_, buffer->buffer, buffer->allocation);

  VkBufferCreateInfo buf_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buf_info.size = size;
  buf_info.usage = usage;

  VmaAllocationCreateInfo alloc_info = {};
  if (mapped) {
    alloc_info.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
  }
  alloc_info.usage = VMA_MEMORY_USAGE_AUTO;

  VmaAllocationInfo info;
  vmaCreateBuffer(allocator_, &buf_info, &alloc_info, &buffer->buffer, &buffer->allocation, &info);

  buffer->usage = usage;
  buffer->size = size;
  if (mapped) buffer->map = reinterpret_cast<uint8_t*>(info.pMappedData);
}

FuchsiaBenchmark::Results FuchsiaBenchmark::Sort(const std::vector<uint32_t>& keys) {
  uint32_t n = static_cast<uint32_t>(keys.size());

  radix_sort_vk_memory_requirements_t mr;
  radix_sort_vk_get_memory_requirements(sorter_keys_, n, &mr);

  constexpr auto kKeyval = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  constexpr auto kInternal = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  Reallocate(&staging_, mr.keyvals_size,
             VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  Reallocate(&keys_even_, mr.keyvals_size, kKeyval);
  Reallocate(&keys_odd_, mr.keyvals_size, kKeyval);
  Reallocate(&keys_internal_, mr.internal_size, kInternal);

  std::memcpy(staging_.map, keys.data(), n * sizeof(uint32_t));

  VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  // upload
  vkBeginCommandBuffer(command_buffer_, &begin);
  VkBufferCopy region = {0, 0, n * sizeof(uint32_t)};
  vkCmdCopyBuffer(command_buffer_, staging_.buffer, keys_even_.buffer, 1, &region);
  vkEndCommandBuffer(command_buffer_);

  VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &command_buffer_;
  vkQueueSubmit(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  vkResetFences(device_, 1, &fence_);

  // sort
  vkBeginCommandBuffer(command_buffer_, &begin);
  vkCmdResetQueryPool(command_buffer_, query_pool_, 0, kTimestampCount);
  vkCmdWriteTimestamp(command_buffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_, 0);

  radix_sort_vk_sort_info_t sort_info = {};
  sort_info.key_bits = 32;
  sort_info.count = n;
  sort_info.keyvals_even = {keys_even_.buffer, 0, mr.keyvals_size};
  sort_info.keyvals_odd = {keys_odd_.buffer, 0, mr.keyvals_size};
  sort_info.internal = {keys_internal_.buffer, 0, mr.internal_size};

  VkDescriptorBufferInfo sorted_out;
  radix_sort_vk_sort(sorter_keys_, &sort_info, device_, command_buffer_, &sorted_out);

  vkCmdWriteTimestamp(command_buffer_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, 1);
  vkEndCommandBuffer(command_buffer_);

  auto cpu_start = std::chrono::steady_clock::now();
  vkQueueSubmit(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  auto cpu_end = std::chrono::steady_clock::now();
  vkResetFences(device_, 1, &fence_);

  // readback from whichever buffer holds the sorted result
  vkBeginCommandBuffer(command_buffer_, &begin);
  region.size = n * sizeof(uint32_t);
  vkCmdCopyBuffer(command_buffer_, sorted_out.buffer, staging_.buffer, 1, &region);
  vkEndCommandBuffer(command_buffer_);
  vkQueueSubmit(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  vkResetFences(device_, 1, &fence_);

  uint64_t timestamps[kTimestampCount];
  vkGetQueryPoolResults(device_, query_pool_, 0, kTimestampCount, sizeof(timestamps), timestamps,
                        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);

  Results result;
  result.keys.resize(n);
  std::memcpy(result.keys.data(), staging_.map, n * sizeof(uint32_t));
  result.total_time = static_cast<uint64_t>((timestamps[1] - timestamps[0]) *
                                            static_cast<double>(timestamp_period_));
  result.cpu_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count();
  return result;
}

FuchsiaBenchmark::Results FuchsiaBenchmark::SortKeyValue(const std::vector<uint32_t>& keys,
                                                         const std::vector<uint32_t>& values) {
  uint32_t n = static_cast<uint32_t>(keys.size());

  radix_sort_vk_memory_requirements_t mr;
  radix_sort_vk_get_memory_requirements(sorter_kv_, n, &mr);

  constexpr auto kKeyval = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  constexpr auto kInternal = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  Reallocate(&staging_, mr.keyvals_size,
             VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  Reallocate(&kv_even_, mr.keyvals_size, kKeyval);
  Reallocate(&kv_odd_, mr.keyvals_size, kKeyval);
  Reallocate(&kv_internal_, mr.internal_size, kInternal);

  // Pack (key << 32 | original_index) to produce a stable sort by key.
  // After sorting, unpack: key = upper 32 bits, value = values[lower 32 bits].
  auto* packed = reinterpret_cast<uint64_t*>(staging_.map);
  for (uint32_t i = 0; i < n; ++i) packed[i] = (static_cast<uint64_t>(keys[i]) << 32) | i;

  VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  // upload
  vkBeginCommandBuffer(command_buffer_, &begin);
  VkBufferCopy region = {0, 0, n * sizeof(uint64_t)};
  vkCmdCopyBuffer(command_buffer_, staging_.buffer, kv_even_.buffer, 1, &region);
  vkEndCommandBuffer(command_buffer_);

  VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &command_buffer_;
  vkQueueSubmit(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  vkResetFences(device_, 1, &fence_);

  // sort
  vkBeginCommandBuffer(command_buffer_, &begin);
  vkCmdResetQueryPool(command_buffer_, query_pool_, 0, kTimestampCount);
  vkCmdWriteTimestamp(command_buffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_, 0);

  radix_sort_vk_sort_info_t sort_info = {};
  sort_info.key_bits = 64;
  sort_info.count = n;
  sort_info.keyvals_even = {kv_even_.buffer, 0, mr.keyvals_size};
  sort_info.keyvals_odd = {kv_odd_.buffer, 0, mr.keyvals_size};
  sort_info.internal = {kv_internal_.buffer, 0, mr.internal_size};

  VkDescriptorBufferInfo sorted_out;
  radix_sort_vk_sort(sorter_kv_, &sort_info, device_, command_buffer_, &sorted_out);

  vkCmdWriteTimestamp(command_buffer_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, 1);
  vkEndCommandBuffer(command_buffer_);

  auto cpu_start = std::chrono::steady_clock::now();
  vkQueueSubmit(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  auto cpu_end = std::chrono::steady_clock::now();
  vkResetFences(device_, 1, &fence_);

  // readback
  vkBeginCommandBuffer(command_buffer_, &begin);
  region.size = n * sizeof(uint64_t);
  vkCmdCopyBuffer(command_buffer_, sorted_out.buffer, staging_.buffer, 1, &region);
  vkEndCommandBuffer(command_buffer_);
  vkQueueSubmit(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  vkResetFences(device_, 1, &fence_);

  uint64_t timestamps[kTimestampCount];
  vkGetQueryPoolResults(device_, query_pool_, 0, kTimestampCount, sizeof(timestamps), timestamps,
                        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);

  Results result;
  result.keys.resize(n);
  result.values.resize(n);
  const auto* sorted = reinterpret_cast<const uint64_t*>(staging_.map);
  for (uint32_t i = 0; i < n; ++i) {
    result.keys[i] = static_cast<uint32_t>(sorted[i] >> 32);
    result.values[i] = values[sorted[i] & 0xFFFFFFFFu];
  }
  result.total_time = static_cast<uint64_t>((timestamps[1] - timestamps[0]) *
                                            static_cast<double>(timestamp_period_));
  result.cpu_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count();
  return result;
}
