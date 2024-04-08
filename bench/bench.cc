#include <iostream>
#include <vector>

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"

#include <vk_radix_sort.h>

namespace {

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

int main() {
  std::cout << "vk_radix_sort benchmark" << std::endl;

  VkInstance instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  uint32_t queue_family_index = 0;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  VmaAllocator allocator = VK_NULL_HANDLE;

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
  messenger_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
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
  vkCreateInstance(&instance_info, NULL, &instance);

  CreateDebugUtilsMessengerEXT(instance, &messenger_info, NULL, &messenger);

  // physical device
  uint32_t physical_device_count = 0;
  vkEnumeratePhysicalDevices(instance, &physical_device_count, NULL);
  std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
  vkEnumeratePhysicalDevices(instance, &physical_device_count,
                             physical_devices.data());
  physical_device = physical_devices[0];

  // find graphics queue
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count,
                                           NULL);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count,
                                           queue_families.data());

  // need just compute and transfer, but add graphics to select a generic queue.
  constexpr VkQueueFlags graphics_queue_flags =
      VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
  for (int i = 0; i < queue_families.size(); ++i) {
    const auto& queue_family = queue_families[i];
    if ((queue_family.queueFlags & graphics_queue_flags) ==
        graphics_queue_flags) {
      queue_family_index = i;
      break;
    }
  }

  // features
  VkPhysicalDeviceSynchronization2Features synchronization_2_features = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};

  VkPhysicalDeviceFeatures2 features = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features.pNext = &synchronization_2_features;
  vkGetPhysicalDeviceFeatures2(physical_device, &features);

  // queues
  std::vector<float> queue_priorities = {
      1.f,
  };
  std::vector<VkDeviceQueueCreateInfo> queue_infos(1);
  queue_infos[0] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue_infos[0].queueFamilyIndex = queue_family_index;
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
  vkCreateDevice(physical_device, &device_info, NULL, &device);

  vkGetDeviceQueue(device, queue_family_index, 0, &queue);

  // vma
  VmaAllocatorCreateInfo allocator_info = {};
  allocator_info.physicalDevice = physical_device;
  allocator_info.device = device;
  allocator_info.instance = instance;
  allocator_info.vulkanApiVersion = VK_API_VERSION_1_3;
  vmaCreateAllocator(&allocator_info, &allocator);

  // commands
  VkCommandPool command_pool;
  VkCommandPoolCreateInfo command_pool_info = {
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  command_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  command_pool_info.queueFamilyIndex = queue_family_index;
  vkCreateCommandPool(device, &command_pool_info, NULL, &command_pool);

  VkCommandBuffer command_buffer;
  VkCommandBufferAllocateInfo command_buffer_info = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  command_buffer_info.commandPool = command_pool;
  command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_buffer_info.commandBufferCount = 1;
  vkAllocateCommandBuffers(device, &command_buffer_info, &command_buffer);

  // fence
  VkFence fence;
  VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  vkCreateFence(device, &fence_info, NULL, &fence);

  // sorter
  VxSorter sorter = VK_NULL_HANDLE;

  VxSorterCreateInfo sorter_info = {};
  sorter_info.device = device;
  vxCreateSorter(&sorter_info, &sorter);

  // commands
  {
    VkCommandBufferBeginInfo command_buffer_begin_info = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    command_buffer_begin_info.flags =
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info);

    int n = 1024;
    // TODO: random init keys
    // TODO: CPU to GPU

    vxCmdRadixSortGlobalHistogram(command_buffer, sorter, n);

    // TODO: GPU to CPU

    vkEndCommandBuffer(command_buffer);

    VkCommandBufferSubmitInfo command_buffer_submit_info = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    command_buffer_submit_info.commandBuffer = command_buffer;

    VkSubmitInfo2 submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &command_buffer_submit_info;
    vkQueueSubmit2(queue, 1, &submit, fence);

    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &fence);
  }

  vkDeviceWaitIdle(device);

  vxDestroySorter(sorter);
  vkDestroyFence(device, fence, NULL);
  vkDestroyCommandPool(device, command_pool, NULL);
  vmaDestroyAllocator(allocator);
  vkDestroyDevice(device, NULL);
  DestroyDebugUtilsMessengerEXT(instance, messenger, NULL);
  vkDestroyInstance(instance, NULL);

  return 0;
}
