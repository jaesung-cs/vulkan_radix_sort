#include <vk_radix_sort.h>

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

#include <shaderc/shaderc.hpp>

#include "generated/upsweep_comp.h"
#include "generated/spine_comp.h"
#include "generated/downsweep_comp.h"
#include "generated/downsweep_key_value_comp.h"

namespace {

constexpr uint32_t RADIX = 256;
constexpr int WORKGROUP_SIZE = 512;
constexpr int PARTITION_DIVISION = 8;
constexpr int PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;

uint32_t RoundUp(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

VkDeviceSize HistogramSize(uint32_t elementCount) {
  return (1 + 4 * RADIX + RoundUp(elementCount, PARTITION_SIZE) * RADIX) *
         sizeof(uint32_t);
}

VkDeviceSize InoutSize(uint32_t elementCount) {
  return elementCount * sizeof(uint32_t);
}

constexpr VkDeviceSize Align(VkDeviceSize offset, VkDeviceSize size) {
  return (offset + size - 1) / size * size;
}

struct StorageOffsets {
  VkDeviceSize elementCountOffset = 0;
  VkDeviceSize histogramOffset = 0;
  VkDeviceSize outOffset = 0;
};

void gpuSort(VkCommandBuffer commandBuffer, VrdxSorter sorter,
             uint32_t elementCount, VkBuffer indirectBuffer,
             VkDeviceSize indirectOffset, VkBuffer buffer, VkDeviceSize offset,
             VkBuffer valueBuffer, VkDeviceSize valueOffset,
             VkQueryPool queryPool, uint32_t query);

void gpuSort(VkCommandBuffer commandBuffer, VrdxSorter sorter,
             uint32_t elementCount, VkBuffer indirectBuffer,
             VkDeviceSize indirectOffset, VkBuffer buffer, VkDeviceSize offset,
             VkBuffer valueBuffer, VkDeviceSize valueOffset,
             VkQueryPool queryPool, uint32_t query);

}  // namespace

struct VrdxSorterLayout_T {
  VkDevice device = VK_NULL_HANDLE;

  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

  VkPipeline upsweepPipeline = VK_NULL_HANDLE;
  VkPipeline spinePipeline = VK_NULL_HANDLE;
  VkPipeline downsweepPipeline = VK_NULL_HANDLE;
  VkPipeline downsweepKeyValuePipeline = VK_NULL_HANDLE;

  uint32_t storageAlignment = 0;
  uint32_t maxWorkgroupSize = 0;
};

struct VrdxSorter_T {
  uint32_t maxElementCount = 0;
  VrdxSorterLayout layout = VK_NULL_HANDLE;
  VmaAllocator allocator = VK_NULL_HANDLE;

  VkBuffer storage = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  StorageOffsets storageOffsets;
};

struct PushConstants {
  uint32_t pass;
  VkDeviceAddress storageReference;
  VkDeviceAddress keysInReference;
  VkDeviceAddress keysOutReference;
  VkDeviceAddress valuesInReference;
  VkDeviceAddress valuesOutReference;
};

void vrdxCreateSorterLayout(const VrdxSorterLayoutCreateInfo* pCreateInfo,
                            VrdxSorterLayout* pSorterLayout) {
  VkDevice device = pCreateInfo->device;
  VkPipelineCache pipelineCache = pCreateInfo->pipelineCache;

  // shader specialization constants and defaults
  VkPhysicalDeviceVulkan11Properties physicalDeviceVulkan11Properties = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
  VkPhysicalDeviceProperties2 physicalDeviceProperties = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  physicalDeviceProperties.pNext = &physicalDeviceVulkan11Properties;

  vkGetPhysicalDeviceProperties2(pCreateInfo->physicalDevice,
                                 &physicalDeviceProperties);

  uint32_t storageAlignment = physicalDeviceProperties.properties.limits
                                  .minStorageBufferOffsetAlignment;
  uint32_t maxWorkgroupSize =
      physicalDeviceProperties.properties.limits.maxComputeWorkGroupSize[0];
  uint32_t subgroupSize = physicalDeviceVulkan11Properties.subgroupSize;

  // pipeline layout
  VkPushConstantRange pushConstants = {};
  pushConstants.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushConstants.offset = 0;
  pushConstants.size = sizeof(PushConstants);

  VkPipelineLayout pipelineLayout;
  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges = &pushConstants;
  vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL, &pipelineLayout);

  // pipelines
  VkPipeline upsweepPipeline;
  {
    VkShaderModule shaderModule;
    VkShaderModuleCreateInfo shaderModuleInfo = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderModuleInfo.codeSize = sizeof(upsweep_comp);
    shaderModuleInfo.pCode = upsweep_comp;
    vkCreateShaderModule(device, &shaderModuleInfo, NULL, &shaderModule);

    VkComputePipelineCreateInfo pipelineInfo = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;

    vkCreateComputePipelines(device, pipelineCache, 1, &pipelineInfo, NULL,
                             &upsweepPipeline);

    vkDestroyShaderModule(device, shaderModule, NULL);
  }

  VkPipeline spinePipeline;
  {
    VkShaderModule shaderModule;
    VkShaderModuleCreateInfo shaderModuleInfo = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderModuleInfo.codeSize = sizeof(spine_comp);
    shaderModuleInfo.pCode = spine_comp;
    vkCreateShaderModule(device, &shaderModuleInfo, NULL, &shaderModule);

    VkComputePipelineCreateInfo pipelineInfo = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;

    vkCreateComputePipelines(device, pipelineCache, 1, &pipelineInfo, NULL,
                             &spinePipeline);

    vkDestroyShaderModule(device, shaderModule, NULL);
  }

  VkPipeline downsweepPipeline;
  VkPipeline downsweepKeyValuePipeline;
  {
    std::vector<VkShaderModule> shaderModules(2);
    VkShaderModuleCreateInfo shaderModuleInfo = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderModuleInfo.codeSize = sizeof(downsweep_comp);
    shaderModuleInfo.pCode = downsweep_comp;
    vkCreateShaderModule(device, &shaderModuleInfo, NULL, &shaderModules[0]);

    shaderModuleInfo.codeSize = sizeof(downsweep_key_value_comp);
    shaderModuleInfo.pCode = downsweep_key_value_comp;
    vkCreateShaderModule(device, &shaderModuleInfo, NULL, &shaderModules[1]);

    std::vector<VkComputePipelineCreateInfo> pipelineInfos(2);
    pipelineInfos[0] = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfos[0].stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfos[0].stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfos[0].stage.module = shaderModules[0];
    pipelineInfos[0].stage.pName = "main";
    pipelineInfos[0].layout = pipelineLayout;

    pipelineInfos[1] = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfos[1].stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfos[1].stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfos[1].stage.module = shaderModules[1];
    pipelineInfos[1].stage.pName = "main";
    pipelineInfos[1].layout = pipelineLayout;

    std::vector<VkPipeline> pipelines(pipelineInfos.size());
    vkCreateComputePipelines(device, pipelineCache, pipelineInfos.size(),
                             pipelineInfos.data(), NULL, pipelines.data());
    downsweepPipeline = pipelines[0];
    downsweepKeyValuePipeline = pipelines[1];

    for (auto shaderModule : shaderModules)
      vkDestroyShaderModule(device, shaderModule, NULL);
  }

  *pSorterLayout = new VrdxSorterLayout_T();
  (*pSorterLayout)->device = device;
  (*pSorterLayout)->pipelineLayout = pipelineLayout;

  (*pSorterLayout)->upsweepPipeline = upsweepPipeline;
  (*pSorterLayout)->spinePipeline = spinePipeline;
  (*pSorterLayout)->downsweepPipeline = downsweepPipeline;
  (*pSorterLayout)->downsweepKeyValuePipeline = downsweepKeyValuePipeline;

  (*pSorterLayout)->storageAlignment = storageAlignment;
  (*pSorterLayout)->maxWorkgroupSize = maxWorkgroupSize;
}

void vrdxDestroySorterLayout(VrdxSorterLayout sorterLayout) {
  vkDestroyPipeline(sorterLayout->device, sorterLayout->upsweepPipeline, NULL);
  vkDestroyPipeline(sorterLayout->device, sorterLayout->spinePipeline, NULL);
  vkDestroyPipeline(sorterLayout->device, sorterLayout->downsweepPipeline,
                    NULL);
  vkDestroyPipeline(sorterLayout->device,
                    sorterLayout->downsweepKeyValuePipeline, NULL);

  vkDestroyPipelineLayout(sorterLayout->device, sorterLayout->pipelineLayout,
                          NULL);
  delete sorterLayout;
}

void vrdxCreateSorter(const VrdxSorterCreateInfo* pCreateInfo,
                      VrdxSorter* pSorter) {
  VrdxSorterLayout sorterLayout = pCreateInfo->sorterLayout;
  VkDevice device = sorterLayout->device;
  uint32_t storageAlignment = sorterLayout->storageAlignment;
  VmaAllocator allocator = pCreateInfo->allocator;
  uint32_t maxElementCount = pCreateInfo->maxElementCount;

  // storage
  VkDeviceSize histogramSize = HistogramSize(maxElementCount);
  // 2x for key value
  VkDeviceSize inoutSize = 2 * InoutSize(maxElementCount);

  // align size from physical device
  StorageOffsets storageOffsets;
  storageOffsets.elementCountOffset = 0;
  storageOffsets.histogramOffset = sizeof(uint32_t);
  storageOffsets.outOffset = Align(histogramSize, storageAlignment);
  VkDeviceSize storageSize =
      Align(storageOffsets.outOffset + inoutSize, storageAlignment);

  VkBuffer storage;
  VmaAllocation allocation;
  {
    VkBufferCreateInfo storageInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    storageInfo.size = storageSize;
    storageInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VmaAllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    vmaCreateBuffer(allocator, &storageInfo, &allocationCreateInfo, &storage,
                    &allocation, NULL);
  }

  *pSorter = new VrdxSorter_T();
  (*pSorter)->maxElementCount = maxElementCount;
  (*pSorter)->layout = sorterLayout;
  (*pSorter)->allocator = allocator;
  (*pSorter)->storage = storage;
  (*pSorter)->allocation = allocation;
  (*pSorter)->storageOffsets = storageOffsets;
}

void vrdxDestroySorter(VrdxSorter sorter) {
  vmaDestroyBuffer(sorter->allocator, sorter->storage, sorter->allocation);
  delete sorter;
}

void vrdxCmdSort(VkCommandBuffer commandBuffer, VrdxSorter sorter,
                 uint32_t elementCount, VkBuffer keysBuffer,
                 VkDeviceSize keysOffset, VkQueryPool queryPool,
                 uint32_t query) {
  gpuSort(commandBuffer, sorter, elementCount, NULL, 0, keysBuffer, keysOffset,
          NULL, 0, queryPool, query);
}

void vrdxCmdSortIndirect(VkCommandBuffer commandBuffer, VrdxSorter sorter,
                         VkBuffer indirectBuffer, VkDeviceSize indirectOffset,
                         VkBuffer keysBuffer, VkDeviceSize keysOffset,
                         VkQueryPool queryPool, uint32_t query) {
  gpuSort(commandBuffer, sorter, 0, indirectBuffer, indirectOffset, keysBuffer,
          keysOffset, NULL, 0, queryPool, query);
}

void vrdxCmdSortKeyValue(VkCommandBuffer commandBuffer, VrdxSorter sorter,
                         uint32_t elementCount, VkBuffer keysBuffer,
                         VkDeviceSize keysOffset, VkBuffer valuesBuffer,
                         VkDeviceSize valuesOffset, VkQueryPool queryPool,
                         uint32_t query) {
  gpuSort(commandBuffer, sorter, elementCount, NULL, 0, keysBuffer, keysOffset,
          valuesBuffer, valuesOffset, queryPool, query);
}

void vrdxCmdSortKeyValueIndirect(VkCommandBuffer commandBuffer,
                                 VrdxSorter sorter, VkBuffer indirectBuffer,
                                 VkDeviceSize indirectOffset,
                                 VkBuffer keysBuffer, VkDeviceSize keysOffset,
                                 VkBuffer valuesBuffer,
                                 VkDeviceSize valuesOffset,
                                 VkQueryPool queryPool, uint32_t query) {
  gpuSort(commandBuffer, sorter, 0, indirectBuffer, indirectOffset, keysBuffer,
          keysOffset, valuesBuffer, valuesOffset, queryPool, query);
}

namespace {

void gpuSort(VkCommandBuffer commandBuffer, VrdxSorter sorter,
             uint32_t elementCount, VkBuffer indirectBuffer,
             VkDeviceSize indirectOffset, VkBuffer keysBuffer,
             VkDeviceSize keysOffset, VkBuffer valuesBuffer,
             VkDeviceSize valuesOffset, VkQueryPool queryPool, uint32_t query) {
  VrdxSorterLayout layout = sorter->layout;
  VkDevice device = layout->device;
  VkBuffer storage = sorter->storage;
  uint32_t maxElementCount = sorter->maxElementCount;
  uint32_t partitionCount =
      RoundUp(indirectBuffer ? maxElementCount : elementCount, PARTITION_SIZE);

  VkDeviceSize histogramSize = HistogramSize(maxElementCount);
  VkDeviceSize inoutSize = InoutSize(maxElementCount);

  const auto& storageOffsets = sorter->storageOffsets;

  if (queryPool) {
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        queryPool, query + 0);
  }

  if (indirectBuffer) {
    // copy elementCount
    VkBufferCopy region;
    region.srcOffset = indirectOffset;
    region.dstOffset = storageOffsets.elementCountOffset;
    region.size = sizeof(uint32_t);
    vkCmdCopyBuffer(commandBuffer, indirectBuffer, storage, 1, &region);
  } else {
    // set element count
    vkCmdUpdateBuffer(commandBuffer, storage, storageOffsets.elementCountOffset,
                      sizeof(elementCount), &elementCount);
  }

  // reset global histogram. partition histogram is set by shader.
  vkCmdFillBuffer(commandBuffer, storage, storageOffsets.histogramOffset,
                  4 * RADIX * sizeof(uint32_t), 0);

  std::vector<VkBufferMemoryBarrier> bufferMemoryBarriers(2);

  bufferMemoryBarriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  bufferMemoryBarriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  bufferMemoryBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  bufferMemoryBarriers[0].buffer = storage;
  bufferMemoryBarriers[0].offset = storageOffsets.elementCountOffset;
  bufferMemoryBarriers[0].size = sizeof(uint32_t);

  bufferMemoryBarriers[1] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  bufferMemoryBarriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  bufferMemoryBarriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  bufferMemoryBarriers[1].buffer = storage;
  bufferMemoryBarriers[1].offset = storageOffsets.histogramOffset;
  bufferMemoryBarriers[1].size = 4 * RADIX * sizeof(uint32_t);

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL,
                       bufferMemoryBarriers.size(), bufferMemoryBarriers.data(),
                       0, NULL);

  if (queryPool) {
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        queryPool, query + 1);
  }

  VkBufferDeviceAddressInfo deviceAddressInfo = {
      VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  deviceAddressInfo.buffer = sorter->storage;
  VkDeviceAddress storageAddress =
      vkGetBufferDeviceAddress(device, &deviceAddressInfo);

  deviceAddressInfo.buffer = keysBuffer;
  VkDeviceAddress keysAddress =
      vkGetBufferDeviceAddress(device, &deviceAddressInfo);

  VkDeviceAddress valuesAddress = 0;
  if (valuesBuffer) {
    deviceAddressInfo.buffer = valuesBuffer;
    valuesAddress = vkGetBufferDeviceAddress(device, &deviceAddressInfo);
  }

  PushConstants pushConstants;
  pushConstants.storageReference =
      storageAddress + storageOffsets.elementCountOffset;

  for (int i = 0; i < 4; ++i) {
    pushConstants.pass = i;
    pushConstants.keysInReference = keysAddress + keysOffset;
    pushConstants.keysOutReference = storageAddress + storageOffsets.outOffset;
    pushConstants.valuesInReference = valuesAddress + valuesOffset;
    pushConstants.valuesOutReference =
        storageAddress + storageOffsets.outOffset + inoutSize;

    if (i % 2 == 1) {
      std::swap(pushConstants.keysInReference, pushConstants.keysOutReference);
      std::swap(pushConstants.valuesInReference,
                pushConstants.valuesOutReference);
    }

    vkCmdPushConstants(commandBuffer, layout->pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                       &pushConstants);

    // upsweep
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      layout->upsweepPipeline);

    vkCmdDispatch(commandBuffer, partitionCount, 1, 1);

    if (queryPool) {
      vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          queryPool, query + 2 + 3 * i + 0);
    }

    // spine
    bufferMemoryBarriers.resize(1);
    bufferMemoryBarriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bufferMemoryBarriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bufferMemoryBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bufferMemoryBarriers[0].buffer = storage;
    bufferMemoryBarriers[0].offset = storageOffsets.histogramOffset;
    bufferMemoryBarriers[0].size = histogramSize;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL,
                         bufferMemoryBarriers.size(),
                         bufferMemoryBarriers.data(), 0, NULL);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      layout->spinePipeline);

    vkCmdDispatch(commandBuffer, RADIX, 1, 1);

    if (queryPool) {
      vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          queryPool, query + 2 + 3 * i + 1);
    }

    // downsweep
    bufferMemoryBarriers.resize(1);
    bufferMemoryBarriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bufferMemoryBarriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bufferMemoryBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bufferMemoryBarriers[0].buffer = storage;
    bufferMemoryBarriers[0].offset = storageOffsets.histogramOffset;
    bufferMemoryBarriers[0].size = histogramSize;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL,
                         bufferMemoryBarriers.size(),
                         bufferMemoryBarriers.data(), 0, NULL);

    if (valuesBuffer) {
      vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        layout->downsweepKeyValuePipeline);
    } else {
      vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        layout->downsweepPipeline);
    }

    vkCmdDispatch(commandBuffer, partitionCount, 1, 1);

    if (queryPool) {
      vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          queryPool, query + 2 + 3 * i + 2);
    }

    if (i < 3) {
      bufferMemoryBarriers.resize(2);

      bufferMemoryBarriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      bufferMemoryBarriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      bufferMemoryBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      bufferMemoryBarriers[0].buffer = storage;
      bufferMemoryBarriers[0].offset = storageOffsets.histogramOffset;
      bufferMemoryBarriers[0].size = histogramSize;

      bufferMemoryBarriers[1] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      bufferMemoryBarriers[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      bufferMemoryBarriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      bufferMemoryBarriers[1].buffer = i % 2 == 0 ? storage : keysBuffer;
      bufferMemoryBarriers[1].offset =
          i % 2 == 0 ? storageOffsets.outOffset : keysOffset;
      bufferMemoryBarriers[1].size = inoutSize;

      if (valuesBuffer) {
        bufferMemoryBarriers.resize(3);
        bufferMemoryBarriers[2] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bufferMemoryBarriers[2].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bufferMemoryBarriers[2].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        bufferMemoryBarriers[2].buffer = i % 2 == 0 ? storage : valuesBuffer;
        bufferMemoryBarriers[2].offset =
            i % 2 == 0 ? storageOffsets.outOffset + inoutSize : valuesOffset;
        bufferMemoryBarriers[2].size = inoutSize;
      }

      vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL,
                           bufferMemoryBarriers.size(),
                           bufferMemoryBarriers.data(), 0, NULL);
    }
  }

  if (queryPool) {
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        queryPool, query + 14);
  }

  // barrier between next command
  {
    bufferMemoryBarriers.resize(1);

    bufferMemoryBarriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bufferMemoryBarriers[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                            VK_ACCESS_SHADER_WRITE_BIT |
                                            VK_ACCESS_TRANSFER_WRITE_BIT;
    bufferMemoryBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                            VK_ACCESS_SHADER_WRITE_BIT |
                                            VK_ACCESS_TRANSFER_WRITE_BIT;
    bufferMemoryBarriers[0].buffer = storage;
    bufferMemoryBarriers[0].offset = 0;
    bufferMemoryBarriers[0].size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, NULL, bufferMemoryBarriers.size(), bufferMemoryBarriers.data(), 0,
        NULL);
  }
}

}  // namespace
