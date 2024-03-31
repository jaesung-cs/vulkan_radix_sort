#ifndef VK_RADIX_SORT_H
#define VK_RADIX_SORT_H

#include <vulkan/vulkan.h>

struct VxSorter_T;

VK_DEFINE_HANDLE(VxSorter)

struct VxSorterCreateInfo {
  VkDevice device;
};

void vxCreateSorter(const VxSorterCreateInfo* pCreateInfo, VxSorter* pSorter);

void vxDestroySorter(VxSorter sorter);

void vxCmdRadixSort(VkCommandBuffer commandBuffer, VxSorter sorter,
                    uint32_t elementCount);

void vxCmdRadixSortGlobalHistogram(VkCommandBuffer commandBuffer,
                                   VxSorter sorter, uint32_t elementCount);

#endif  // VK_RADIX_SORT_H
