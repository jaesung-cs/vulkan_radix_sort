#include "vulkan_benchmark.h"

#include <iostream>

namespace {

constexpr auto timestamp_count = 15;

static VKAPI_ATTR VkBool32 VKAPI_CALL
DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
              void* pUserData) {
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl
            << std::endl;

  return VK_FALSE;
}

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
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

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks* pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

}  // namespace

VulkanBenchmark::VulkanBenchmark() {
  // instance
  VkApplicationInfo application_info = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
  application_info.pApplicationName = "vk_radix_sort_benchmark";
  application_info.applicationVersion = VK_MAKE_API_VERSION(0, 0, 0, 0);
  application_info.pEngineName = "vk_radix_sort";
  application_info.engineVersion = VK_MAKE_API_VERSION(0, 0, 0, 0);
  application_info.apiVersion = VK_API_VERSION_1_2;

  VkDebugUtilsMessengerCreateInfoEXT messenger_info = {
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
  messenger_info.messageSeverity =
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  messenger_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  messenger_info.pfnUserCallback = DebugCallback;

  std::vector<const char*> layers = {"VK_LAYER_KHRONOS_validation"};
  std::vector<const char*> instance_extensions = {
      VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
#ifdef __APPLE__
      VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
#endif
  };

  VkInstanceCreateInfo instance_info = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
#ifdef __APPLE__
  instance_info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
  instance_info.pNext = &messenger_info;
  instance_info.pApplicationInfo = &application_info;
  instance_info.enabledLayerCount = layers.size();
  instance_info.ppEnabledLayerNames = layers.data();
  instance_info.enabledExtensionCount = instance_extensions.size();
  instance_info.ppEnabledExtensionNames = instance_extensions.data();
  vkCreateInstance(&instance_info, NULL, &instance_);

  CreateDebugUtilsMessengerEXT(instance_, &messenger_info, NULL, &messenger_);

  // physical device
  uint32_t physical_device_count = 0;
  vkEnumeratePhysicalDevices(instance_, &physical_device_count, NULL);
  std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
  vkEnumeratePhysicalDevices(instance_, &physical_device_count,
                             physical_devices.data());
  physical_device_ = physical_devices[0];

  // find graphics queue
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_,
                                           &queue_family_count, NULL);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device_, &queue_family_count, queue_families.data());

  for (int i = 0; i < queue_families.size(); ++i) {
    const auto& queue_family = queue_families[i];
    if ((queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT) ==
            VK_QUEUE_COMPUTE_BIT &&
        (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
      queue_family_index_ = i;
      break;
    }
  }

  // features
  VkPhysicalDeviceBufferDeviceAddressFeatures buffer_device_address_features = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};

  VkPhysicalDeviceFeatures2 features = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features.pNext = &buffer_device_address_features;
  vkGetPhysicalDeviceFeatures2(physical_device_, &features);

  // queues
  std::vector<float> queue_priorities = {
      1.f,
  };
  std::vector<VkDeviceQueueCreateInfo> queue_infos(1);
  queue_infos[0] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue_infos[0].queueFamilyIndex = queue_family_index_;
  queue_infos[0].queueCount = queue_priorities.size();
  queue_infos[0].pQueuePriorities = queue_priorities.data();

  std::vector<const char*> device_extensions = {
#ifdef __APPLE__
      "VK_KHR_portability_subset",
#endif
  };

  VkDeviceCreateInfo device_info = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device_info.pNext = &features;
  device_info.queueCreateInfoCount = queue_infos.size();
  device_info.pQueueCreateInfos = queue_infos.data();
  device_info.enabledExtensionCount = device_extensions.size();
  device_info.ppEnabledExtensionNames = device_extensions.data();
  vkCreateDevice(physical_device_, &device_info, NULL, &device_);

  vkGetDeviceQueue(device_, queue_family_index_, 0, &queue_);

  // vma
  VmaAllocatorCreateInfo allocator_info = {};
  allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  allocator_info.physicalDevice = physical_device_;
  allocator_info.device = device_;
  allocator_info.instance = instance_;
  allocator_info.vulkanApiVersion = application_info.apiVersion;
  vmaCreateAllocator(&allocator_info, &allocator_);

  // commands
  VkCommandPoolCreateInfo command_pool_info = {
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                            VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  command_pool_info.queueFamilyIndex = queue_family_index_;
  vkCreateCommandPool(device_, &command_pool_info, NULL, &command_pool_);

  VkCommandBufferAllocateInfo command_buffer_info = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  command_buffer_info.commandPool = command_pool_;
  command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_buffer_info.commandBufferCount = 1;
  vkAllocateCommandBuffers(device_, &command_buffer_info, &command_buffer_);

  // fence
  VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  vkCreateFence(device_, &fence_info, NULL, &fence_);

  // timestamp query pool
  VkQueryPoolCreateInfo query_pool_info = {
      VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  query_pool_info.queryCount = timestamp_count;
  vkCreateQueryPool(device_, &query_pool_info, NULL, &query_pool_);

  // sorter
  VrdxSorterCreateInfo sorter_info = {};
  sorter_info.physicalDevice = physical_device_;
  sorter_info.device = device_;
  vrdxCreateSorter(&sorter_info, &sorter_);
}

VulkanBenchmark::~VulkanBenchmark() {
  vkDeviceWaitIdle(device_);

  if (keys_.buffer)
    vmaDestroyBuffer(allocator_, keys_.buffer, keys_.allocation);
  if (storage_.buffer)
    vmaDestroyBuffer(allocator_, storage_.buffer, storage_.allocation);
  if (staging_.buffer)
    vmaDestroyBuffer(allocator_, staging_.buffer, staging_.allocation);

  vrdxDestroySorter(sorter_);
  vkDestroyQueryPool(device_, query_pool_, NULL);
  vkDestroyFence(device_, fence_, NULL);
  vkDestroyCommandPool(device_, command_pool_, NULL);
  vmaDestroyAllocator(allocator_);
  vkDestroyDevice(device_, NULL);
  DestroyDebugUtilsMessengerEXT(instance_, messenger_, NULL);
  vkDestroyInstance(instance_, NULL);
}

void VulkanBenchmark::Reallocate(Buffer* buffer, VkDeviceSize size,
                                 VkBufferUsageFlags usage, bool mapped) {
  if ((buffer->usage & usage) == usage && buffer->size >= size &&
      (mapped && buffer->map || !mapped && buffer->map == nullptr))
    return;

  if (buffer->allocation)
    vmaDestroyBuffer(allocator_, buffer->buffer, buffer->allocation);

  VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buffer_info.size = size;
  buffer_info.usage = usage;
  VmaAllocationCreateInfo allocation_create_info = {};
  if (mapped) {
    allocation_create_info.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;
  }
  allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;

  VmaAllocationInfo allocation_info;
  vmaCreateBuffer(allocator_, &buffer_info, &allocation_create_info,
                  &buffer->buffer, &buffer->allocation, &allocation_info);

  buffer->usage = usage;
  buffer->size = size;
  if (mapped)
    buffer->map = reinterpret_cast<uint8_t*>(allocation_info.pMappedData);
}

VulkanBenchmark::Results VulkanBenchmark::Sort(
    const std::vector<uint32_t>& keys) {
  auto element_count = keys.size();

  Reallocate(
      &staging_, element_count * sizeof(uint32_t),
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      true);
  Reallocate(&keys_, element_count * sizeof(uint32_t),
             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  VrdxSorterStorageRequirements requirements;
  vrdxGetSorterStorageRequirements(sorter_, element_count, &requirements);
  Reallocate(&storage_, requirements.size, requirements.usage);

  std::memcpy(staging_.map, keys.data(), element_count * sizeof(uint32_t));

  VkCommandBufferBeginInfo command_buffer_begin_info = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(command_buffer_, &command_buffer_begin_info);

  vkCmdResetQueryPool(command_buffer_, query_pool_, 0, timestamp_count);

  // copy to keys buffer
  VkBufferCopy region = {};
  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size = element_count * sizeof(uint32_t);
  vkCmdCopyBuffer(command_buffer_, staging_.buffer, keys_.buffer, 1, &region);

  vkEndCommandBuffer(command_buffer_);

  VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &command_buffer_;
  vkQueueSubmit(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  vkResetFences(device_, 1, &fence_);

  // sort
  vkBeginCommandBuffer(command_buffer_, &command_buffer_begin_info);

  vrdxCmdSort(command_buffer_, sorter_, element_count, keys_.buffer, 0,
              storage_.buffer, 0, query_pool_, 0);

  vkEndCommandBuffer(command_buffer_);
  vkQueueSubmit(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  vkResetFences(device_, 1, &fence_);

  // copy back
  vkBeginCommandBuffer(command_buffer_, &command_buffer_begin_info);

  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size = element_count * sizeof(uint32_t);
  vkCmdCopyBuffer(command_buffer_, keys_.buffer, staging_.buffer, 1, &region);

  vkEndCommandBuffer(command_buffer_);
  vkQueueSubmit(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  vkResetFences(device_, 1, &fence_);

  std::vector<uint64_t> timestamps(timestamp_count);
  vkGetQueryPoolResults(device_, query_pool_, 0, timestamps.size(),
                        timestamps.size() * sizeof(uint64_t), timestamps.data(),
                        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);

  Results result;
  result.keys.resize(element_count);
  std::memcpy(result.keys.data(), staging_.map,
              element_count * sizeof(uint32_t));
  result.total_time = timestamps[timestamp_count - 1] - timestamps[0];
  return result;
}

VulkanBenchmark::Results VulkanBenchmark::SortKeyValue(
    const std::vector<uint32_t>& keys, const std::vector<uint32_t>& values) {
  auto element_count = keys.size();

  Reallocate(
      &staging_, (2 * element_count + 1) * sizeof(uint32_t),
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      true);
  Reallocate(&keys_, (2 * element_count + 1) * sizeof(uint32_t),
             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  VrdxSorterStorageRequirements requirements;
  vrdxGetSorterKeyValueStorageRequirements(sorter_, element_count,
                                           &requirements);
  Reallocate(&storage_, requirements.size, requirements.usage);

  std::memcpy(staging_.map, keys.data(), element_count * sizeof(uint32_t));
  std::memcpy(staging_.map + element_count * sizeof(uint32_t), values.data(),
              element_count * sizeof(uint32_t));
  std::memcpy(staging_.map + 2 * element_count * sizeof(uint32_t),
              &element_count, sizeof(uint32_t));

  VkCommandBufferBeginInfo command_buffer_begin_info = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(command_buffer_, &command_buffer_begin_info);

  vkCmdResetQueryPool(command_buffer_, query_pool_, 0, timestamp_count);

  // copy to keys buffer
  VkBufferCopy region = {};
  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size = (2 * element_count + 1) * sizeof(uint32_t);
  vkCmdCopyBuffer(command_buffer_, staging_.buffer, keys_.buffer, 1, &region);

  vkEndCommandBuffer(command_buffer_);

  VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &command_buffer_;
  vkQueueSubmit(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  vkResetFences(device_, 1, &fence_);

  // sort
  vkBeginCommandBuffer(command_buffer_, &command_buffer_begin_info);

  vrdxCmdSortKeyValueIndirect(
      command_buffer_, sorter_, element_count, keys_.buffer,
      2 * element_count * sizeof(uint32_t), keys_.buffer, 0, keys_.buffer,
      element_count * sizeof(uint32_t), storage_.buffer, 0, query_pool_, 0);

  vkEndCommandBuffer(command_buffer_);
  vkQueueSubmit(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  vkResetFences(device_, 1, &fence_);

  // copy back
  vkBeginCommandBuffer(command_buffer_, &command_buffer_begin_info);

  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size = 2 * element_count * sizeof(uint32_t);
  vkCmdCopyBuffer(command_buffer_, keys_.buffer, staging_.buffer, 1, &region);

  vkEndCommandBuffer(command_buffer_);
  vkQueueSubmit(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  vkResetFences(device_, 1, &fence_);

  std::vector<uint64_t> timestamps(timestamp_count);
  vkGetQueryPoolResults(device_, query_pool_, 0, timestamps.size(),
                        timestamps.size() * sizeof(uint64_t), timestamps.data(),
                        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);

  Results result;
  result.keys.resize(element_count);
  result.values.resize(element_count);
  std::memcpy(result.keys.data(), staging_.map,
              element_count * sizeof(uint32_t));
  std::memcpy(result.values.data(),
              staging_.map + element_count * sizeof(uint32_t),
              element_count * sizeof(uint32_t));
  result.total_time = timestamps[timestamp_count - 1] - timestamps[0];
  return result;
}
