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

  // need just compute and transfer, but add graphics to select a generic queue.
  constexpr VkQueueFlags graphics_queue_flags =
      VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
  for (int i = 0; i < queue_families.size(); ++i) {
    const auto& queue_family = queue_families[i];
    if ((queue_family.queueFlags & graphics_queue_flags) ==
        graphics_queue_flags) {
      queue_family_index_ = i;
      break;
    }
  }

  // features
  VkPhysicalDeviceSynchronization2Features synchronization_2_features = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};

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

  // sorter
  VxSorterCreateInfo sorter_info = {};
  sorter_info.device = device_;
  sorter_info.maxCommandsInFlight = 6;
  vxCreateSorter(&sorter_info, &sorter_);

  // preallocate buffers
  {
    VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.size = MAX_ELEMENT_COUNT * sizeof(uint32_t);
    VmaAllocationCreateInfo allocation_create_info = {};
    allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    vmaCreateBuffer(allocator_, &buffer_info, &allocation_create_info,
                    &keys_.buffer, &keys_.allocation, NULL);
  }
  {
    VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.size = MAX_ELEMENT_COUNT * sizeof(uint32_t);
    VmaAllocationCreateInfo allocation_create_info = {};
    allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    vmaCreateBuffer(allocator_, &buffer_info, &allocation_create_info,
                    &out_keys_.buffer, &out_keys_.allocation, NULL);
  }
  {
    VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.size = 2 * 4 * 256 * sizeof(uint32_t);
    VmaAllocationCreateInfo allocation_create_info = {};
    allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    vmaCreateBuffer(allocator_, &buffer_info, &allocation_create_info,
                    &histogram_.buffer, &histogram_.allocation, NULL);
  }
  {
    VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.size = (1 + RADIX * ((MAX_ELEMENT_COUNT + PARTITION_SIZE - 1) /
                                     PARTITION_SIZE)) *
                       sizeof(uint32_t);
    VmaAllocationCreateInfo allocation_create_info = {};
    allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    vmaCreateBuffer(allocator_, &buffer_info, &allocation_create_info,
                    &lookback_.buffer, &lookback_.allocation, NULL);
  }
  {
    VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.size = MAX_ELEMENT_COUNT * sizeof(uint32_t);
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
  vmaDestroyBuffer(allocator_, out_keys_.buffer, out_keys_.allocation);
  vmaDestroyBuffer(allocator_, histogram_.buffer, histogram_.allocation);
  vmaDestroyBuffer(allocator_, lookback_.buffer, lookback_.allocation);
  vmaDestroyBuffer(allocator_, staging_.buffer, staging_.allocation);

  vxDestroySorter(sorter_);
  vkDestroyFence(device_, fence_, NULL);
  vkDestroyCommandPool(device_, command_pool_, NULL);
  vmaDestroyAllocator(allocator_);
  vkDestroyDevice(device_, NULL);
  DestroyDebugUtilsMessengerEXT(instance_, messenger_, NULL);
  vkDestroyInstance(instance_, NULL);
}

VulkanBenchmarkBase::IntermediateResults VulkanBenchmarkBase::GlobalHistogram(
    const std::vector<uint32_t>& keys) {
  auto element_count = keys.size();

  std::memcpy(staging_.map, keys.data(), element_count * sizeof(uint32_t));

  VkCommandBufferBeginInfo command_buffer_begin_info = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(command_buffer_, &command_buffer_begin_info);

  // copy to keys buffer
  VkBufferCopy region = {};
  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size = element_count * sizeof(uint32_t);
  vkCmdCopyBuffer(command_buffer_, staging_.buffer, keys_.buffer, 1, &region);

  // histogram
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

  vxCmdRadixSortGlobalHistogram(command_buffer_, sorter_, element_count,
                                keys_.buffer, 0, histogram_.buffer, 0);

  // scan
  buffer_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
  buffer_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  buffer_barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
  buffer_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  buffer_barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
  buffer_barrier.buffer = histogram_.buffer;
  buffer_barrier.offset = 0;
  buffer_barrier.size = 4 * RADIX * sizeof(uint32_t);
  vkCmdPipelineBarrier2(command_buffer_, &dependency_info);

  vxCmdRadixSortGlobalHistogramScan(command_buffer_, sorter_, histogram_.buffer,
                                    0, histogram_.buffer,
                                    4 * RADIX * sizeof(uint32_t));

  // copy to staging buffer
  buffer_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
  buffer_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  buffer_barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
  buffer_barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  buffer_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
  buffer_barrier.buffer = histogram_.buffer;
  buffer_barrier.offset = 0;
  buffer_barrier.size = 2 * 4 * RADIX * sizeof(uint32_t);
  vkCmdPipelineBarrier2(command_buffer_, &dependency_info);

  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size = 2 * 4 * RADIX * sizeof(uint32_t);
  vkCmdCopyBuffer(command_buffer_, histogram_.buffer, staging_.buffer, 1,
                  &region);

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

  IntermediateResults result;
  result.histogram.resize(4 * RADIX);
  std::memcpy(result.histogram.data(), staging_.map,
              4 * RADIX * sizeof(uint32_t));
  result.histogram_cumsum.resize(4 * RADIX);
  std::memcpy(result.histogram_cumsum.data(),
              staging_.map + 4 * RADIX * sizeof(uint32_t),
              4 * RADIX * sizeof(uint32_t));
  return result;
}

VulkanBenchmarkBase::IntermediateResults VulkanBenchmarkBase::SortSteps(
    const std::vector<uint32_t>& keys) {
  auto element_count = keys.size();
  VkDeviceSize histogram_size = 4 * RADIX * sizeof(uint32_t);

  auto result = GlobalHistogram(keys);

  // now histogram buffer remains in GPU. binning.
  for (int i = 0; i < 4; i++) {
    std::cout << "sort pass " << i << std::endl;

    VkBuffer in, out;
    if (i % 2 == 0) {
      in = keys_.buffer;
      out = out_keys_.buffer;
    } else {
      in = out_keys_.buffer;
      out = keys_.buffer;
    }

    VkCommandBufferBeginInfo command_buffer_begin_info = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    command_buffer_begin_info.flags =
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(command_buffer_, &command_buffer_begin_info);

    vxCmdRadixSortBinning(command_buffer_, sorter_, element_count, i, in, 0,
                          histogram_.buffer, histogram_size, lookback_.buffer,
                          0, out, 0);

    // copy to staging buffer
    VkBufferMemoryBarrier2 buffer_barrier = {
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    buffer_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    buffer_barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    buffer_barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    buffer_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    buffer_barrier.buffer = out;
    buffer_barrier.offset = 0;
    buffer_barrier.size = element_count * sizeof(uint32_t);
    VkDependencyInfo dependency_info = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency_info.bufferMemoryBarrierCount = 1;
    dependency_info.pBufferMemoryBarriers = &buffer_barrier;
    vkCmdPipelineBarrier2(command_buffer_, &dependency_info);

    VkBufferCopy region;
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = element_count * sizeof(uint32_t);
    vkCmdCopyBuffer(command_buffer_, out, staging_.buffer, 1, &region);

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

    result.keys[i].resize(element_count);
    std::memcpy(result.keys[i].data(), staging_.map,
                element_count * sizeof(uint32_t));
  }

  return result;
}

VulkanBenchmarkBase::IntermediateResults VulkanBenchmarkBase::Sort(
    const std::vector<uint32_t>& keys) {
  auto element_count = keys.size();

  std::memcpy(staging_.map, keys.data(), element_count * sizeof(uint32_t));

  VkCommandBufferBeginInfo command_buffer_begin_info = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(command_buffer_, &command_buffer_begin_info);

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

  vxCmdRadixSort(command_buffer_, sorter_, element_count, keys_.buffer, 0,
                 histogram_.buffer, 0, histogram_.buffer,
                 4 * RADIX * sizeof(uint32_t), lookback_.buffer, 0,
                 out_keys_.buffer, 0);

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

  IntermediateResults result;
  result.keys[3].resize(element_count);
  std::memcpy(result.keys[3].data(), staging_.map,
              element_count * sizeof(uint32_t));
  return result;
}
