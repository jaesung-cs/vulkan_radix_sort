#include "vulkan_benchmark_base.h"

#include <iostream>
#include <vector>

namespace {

constexpr uint32_t RADIX = 256;

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

VulkanBenchmarkBase::VulkanBenchmarkBase() {
  // instance
  VkApplicationInfo application_info = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
  application_info.pApplicationName = "vk_radix_sort_benchmark";
  application_info.applicationVersion = VK_MAKE_API_VERSION(0, 0, 0, 0);
  application_info.pEngineName = "vk_radix_sort";
  application_info.engineVersion = VK_MAKE_API_VERSION(0, 0, 0, 0);
  application_info.apiVersion = VK_API_VERSION_1_3;

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
      VK_EXT_DEBUG_UTILS_EXTENSION_NAME};

  VkInstanceCreateInfo instance_info = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
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
  VkPhysicalDeviceMaintenance4Features maintenance_4_features = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES};

  VkPhysicalDeviceSynchronization2Features synchronization_2_features = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
  synchronization_2_features.pNext = &maintenance_4_features;

  VkPhysicalDeviceFeatures2 features = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features.pNext = &synchronization_2_features;
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
      VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
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
  allocator_info.physicalDevice = physical_device_;
  allocator_info.device = device_;
  allocator_info.instance = instance_;
  allocator_info.vulkanApiVersion = VK_API_VERSION_1_3;
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
  query_pool_info.queryCount = 15;
  vkCreateQueryPool(device_, &query_pool_info, NULL, &query_pool_);

  // sorter
  VrdxSorterLayoutCreateInfo sorter_layout_info = {};
  sorter_layout_info.physicalDevice = physical_device_;
  sorter_layout_info.device = device_;
  vrdxCreateSorterLayout(&sorter_layout_info, &sorter_layout_);

  VrdxSorterCreateInfo sorter_info = {};
  sorter_info.allocator = allocator_;
  sorter_info.sorterLayout = sorter_layout_;
  sorter_info.maxElementCount = 10000000;
  sorter_info.maxCommandsInFlight = 1;
  vrdxCreateSorter(&sorter_info, &sorter_);

  // preallocate buffers
  {
    VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.size = (2 * MAX_ELEMENT_COUNT + 1) * sizeof(uint32_t);
    VmaAllocationCreateInfo allocation_create_info = {};
    allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    vmaCreateBuffer(allocator_, &buffer_info, &allocation_create_info,
                    &keys_.buffer, &keys_.allocation, NULL);
  }
  {
    VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.size = (2 * MAX_ELEMENT_COUNT + 1) * sizeof(uint32_t);
    VmaAllocationCreateInfo allocation_create_info = {};
    allocation_create_info.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;
    allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    VmaAllocationInfo allocation_info;
    vmaCreateBuffer(allocator_, &buffer_info, &allocation_create_info,
                    &staging_.buffer, &staging_.allocation, &allocation_info);
    staging_.map = reinterpret_cast<uint8_t*>(allocation_info.pMappedData);
  }
}

VulkanBenchmarkBase::~VulkanBenchmarkBase() {
  vkDeviceWaitIdle(device_);

  vmaDestroyBuffer(allocator_, keys_.buffer, keys_.allocation);
  vmaDestroyBuffer(allocator_, staging_.buffer, staging_.allocation);

  vrdxDestroySorter(sorter_);
  vrdxDestroySorterLayout(sorter_layout_);
  vkDestroyQueryPool(device_, query_pool_, NULL);
  vkDestroyFence(device_, fence_, NULL);
  vkDestroyCommandPool(device_, command_pool_, NULL);
  vmaDestroyAllocator(allocator_);
  vkDestroyDevice(device_, NULL);
  DestroyDebugUtilsMessengerEXT(instance_, messenger_, NULL);
  vkDestroyInstance(instance_, NULL);
}

VulkanBenchmarkBase::IntermediateResults VulkanBenchmarkBase::Sort(
    const std::vector<uint32_t>& keys) {
  constexpr auto sort_method = VRDX_SORT_METHOD_REDUCE_THEN_SCAN;
  const auto timestamp_count =
      sort_method == VRDX_SORT_METHOD_ONESWEEP ? 8 : 15;

  auto element_count = keys.size();

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

  // sort
  VkBufferMemoryBarrier2 buffer_barrier = {
      VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
  buffer_barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  buffer_barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  buffer_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  buffer_barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
  buffer_barrier.buffer = keys_.buffer;
  buffer_barrier.offset = 0;
  buffer_barrier.size = element_count * sizeof(uint32_t);
  VkDependencyInfo dependency_info = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency_info.bufferMemoryBarrierCount = 1;
  dependency_info.pBufferMemoryBarriers = &buffer_barrier;
  vkCmdPipelineBarrier2(command_buffer_, &dependency_info);

  vrdxCmdSort(command_buffer_, sorter_, sort_method, element_count,
              keys_.buffer, 0, query_pool_, 0);

  // copy back
  buffer_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
  buffer_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  buffer_barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
  buffer_barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  buffer_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
  buffer_barrier.buffer = keys_.buffer;
  buffer_barrier.offset = 0;
  buffer_barrier.size = element_count * sizeof(uint32_t);
  dependency_info = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency_info.bufferMemoryBarrierCount = 1;
  dependency_info.pBufferMemoryBarriers = &buffer_barrier;
  vkCmdPipelineBarrier2(command_buffer_, &dependency_info);

  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size = element_count * sizeof(uint32_t);
  vkCmdCopyBuffer(command_buffer_, keys_.buffer, staging_.buffer, 1, &region);

  vkEndCommandBuffer(command_buffer_);

  VkCommandBufferSubmitInfo command_buffer_submit_info = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  command_buffer_submit_info.commandBuffer = command_buffer_;
  VkSubmitInfo2 submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  submit.commandBufferInfoCount = 1;
  submit.pCommandBufferInfos = &command_buffer_submit_info;
  vkQueueSubmit2(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  vkResetFences(device_, 1, &fence_);

  std::vector<uint64_t> timestamps(timestamp_count);
  vkGetQueryPoolResults(device_, query_pool_, 0, timestamps.size(),
                        timestamps.size() * sizeof(uint64_t), timestamps.data(),
                        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);

  IntermediateResults result;
  result.keys[3].resize(element_count);
  std::memcpy(result.keys[3].data(), staging_.map,
              element_count * sizeof(uint32_t));
  result.total_time = timestamps[timestamp_count - 1] - timestamps[0];
  result.binning_times.resize(4);
  if (sort_method == VRDX_SORT_METHOD_ONESWEEP) {
    result.histogram_time = timestamps[1] - timestamps[0];
    result.scan_time = timestamps[2] - timestamps[1];
    result.binning_times[0] = timestamps[3] - timestamps[2];
    result.binning_times[1] = timestamps[4] - timestamps[3];
    result.binning_times[2] = timestamps[5] - timestamps[4];
    result.binning_times[3] = timestamps[6] - timestamps[5];
  } else if (sort_method == VRDX_SORT_METHOD_REDUCE_THEN_SCAN) {
    result.reduce_then_scan_times.resize(14);
    for (int i = 0; i < 14; ++i) {
      result.reduce_then_scan_times[i] = timestamps[i + 1] - timestamps[i];
    }
  }
  return result;
}

VulkanBenchmarkBase::IntermediateResults VulkanBenchmarkBase::SortKeyValue(
    const std::vector<uint32_t>& keys, const std::vector<uint32_t>& values) {
  constexpr auto sort_method = VRDX_SORT_METHOD_REDUCE_THEN_SCAN;
  const auto timestamp_count =
      sort_method == VRDX_SORT_METHOD_ONESWEEP ? 8 : 15;

  auto element_count = keys.size();

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

  // sort
  VkBufferMemoryBarrier2 buffer_barrier = {
      VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
  buffer_barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  buffer_barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  buffer_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  buffer_barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
  buffer_barrier.buffer = keys_.buffer;
  buffer_barrier.offset = 0;
  buffer_barrier.size = (2 * element_count + 1) * sizeof(uint32_t);
  VkDependencyInfo dependency_info = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency_info.bufferMemoryBarrierCount = 1;
  dependency_info.pBufferMemoryBarriers = &buffer_barrier;
  vkCmdPipelineBarrier2(command_buffer_, &dependency_info);

  vrdxCmdSortKeyValueIndirect(
      command_buffer_, sorter_, sort_method, keys_.buffer,
      2 * element_count * sizeof(uint32_t), keys_.buffer, 0, keys_.buffer,
      element_count * sizeof(uint32_t), query_pool_, 0);

  // copy back
  buffer_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
  buffer_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  buffer_barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
  buffer_barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  buffer_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
  buffer_barrier.buffer = keys_.buffer;
  buffer_barrier.offset = 0;
  buffer_barrier.size = 2 * element_count * sizeof(uint32_t);
  dependency_info = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency_info.bufferMemoryBarrierCount = 1;
  dependency_info.pBufferMemoryBarriers = &buffer_barrier;
  vkCmdPipelineBarrier2(command_buffer_, &dependency_info);

  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size = 2 * element_count * sizeof(uint32_t);
  vkCmdCopyBuffer(command_buffer_, keys_.buffer, staging_.buffer, 1, &region);

  vkEndCommandBuffer(command_buffer_);

  VkCommandBufferSubmitInfo command_buffer_submit_info = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  command_buffer_submit_info.commandBuffer = command_buffer_;
  VkSubmitInfo2 submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  submit.commandBufferInfoCount = 1;
  submit.pCommandBufferInfos = &command_buffer_submit_info;
  vkQueueSubmit2(queue_, 1, &submit, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  vkResetFences(device_, 1, &fence_);

  std::vector<uint64_t> timestamps(timestamp_count);
  vkGetQueryPoolResults(device_, query_pool_, 0, timestamps.size(),
                        timestamps.size() * sizeof(uint64_t), timestamps.data(),
                        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);

  IntermediateResults result;
  result.keys[3].resize(element_count);
  result.values.resize(element_count);
  std::memcpy(result.keys[3].data(), staging_.map,
              element_count * sizeof(uint32_t));
  std::memcpy(result.values.data(),
              staging_.map + element_count * sizeof(uint32_t),
              element_count * sizeof(uint32_t));
  result.total_time = timestamps[timestamp_count - 1] - timestamps[0];
  result.binning_times.resize(4);
  if (sort_method == VRDX_SORT_METHOD_ONESWEEP) {
    result.histogram_time = timestamps[1] - timestamps[0];
    result.scan_time = timestamps[2] - timestamps[1];
    result.binning_times[0] = timestamps[3] - timestamps[2];
    result.binning_times[1] = timestamps[4] - timestamps[3];
    result.binning_times[2] = timestamps[5] - timestamps[4];
    result.binning_times[3] = timestamps[6] - timestamps[5];
  } else if (sort_method == VRDX_SORT_METHOD_REDUCE_THEN_SCAN) {
    result.reduce_then_scan_times.resize(14);
    for (int i = 0; i < 14; ++i) {
      result.reduce_then_scan_times[i] = timestamps[i + 1] - timestamps[i];
    }
  }
  return result;
}
