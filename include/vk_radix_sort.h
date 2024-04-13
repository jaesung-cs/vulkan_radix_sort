#ifndef VK_RADIX_SORT_H
#define VK_RADIX_SORT_H

#include <vulkan/vulkan.h>

struct VxSorter_T;

VK_DEFINE_HANDLE(VxSorter)

struct VxSorterCreateInfo {
  VkDevice device;
  uint32_t maxCommandsInFlight;
};

void vxCreateSorter(const VxSorterCreateInfo* pCreateInfo, VxSorter* pSorter);

void vxDestroySorter(VxSorter sorter);

void vxGetSorterBufferSize(uint32_t maxElementCount, VkDeviceSize* size);

void vxCmdRadixSort(VkCommandBuffer commandBuffer, VxSorter sorter,
                    uint32_t elementCount);

void vxCmdRadixSortGlobalHistogram(VkCommandBuffer commandBuffer,
                                   VxSorter sorter, uint32_t elementCount,
                                   VkBuffer buffer, VkDeviceSize offset,
                                   VkBuffer histogramBuffer,
                                   VkDeviceSize histogramOffset);

#endif  // VK_RADIX_SORT_H
