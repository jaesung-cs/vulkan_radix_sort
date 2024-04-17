#include <vk_radix_sort.h>

#include <iostream>
#include <vector>
#include <string>

#include <shaderc/shaderc.hpp>

#include "shader/histogram.h"
#include "shader/scan.h"
#include "shader/binning.h"

namespace {

constexpr uint32_t RADIX = 256;
constexpr uint32_t PARTITION_SIZE = 7680;

uint32_t RoundUp(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

constexpr VkDeviceSize HistogramSize() { return 4 * RADIX * sizeof(uint32_t); }

VkDeviceSize InoutSize(uint32_t elementCount) {
  return elementCount * sizeof(uint32_t);
}

VkDeviceSize LookbackSize(uint32_t elementCount) {
  return (1 + RADIX * RoundUp(elementCount, PARTITION_SIZE)) * sizeof(uint32_t);
}

constexpr VkDeviceSize Align(VkDeviceSize offset, VkDeviceSize size) {
  return (offset + size - 1) / size * size;
}

struct StorageOffsets {
  VkDeviceSize histogramOffset = 0;
  VkDeviceSize lookbackOffset = 0;
  VkDeviceSize outOffset = 0;
};

// Compiles a shader to a SPIR-V binary, and create a VkShaderModule.
VkShaderModule CreateShaderModule(VkDevice device, VkShaderStageFlagBits stage,
                                  const std::string& source) {
  shaderc_shader_kind kind;
  switch (stage) {
    case VK_SHADER_STAGE_VERTEX_BIT:
      kind = shaderc_glsl_vertex_shader;
      break;

    case VK_SHADER_STAGE_FRAGMENT_BIT:
      kind = shaderc_glsl_fragment_shader;
      break;

    case VK_SHADER_STAGE_COMPUTE_BIT:
      kind = shaderc_glsl_compute_shader;
      break;
  }

  shaderc::Compiler compiler;
  shaderc::CompileOptions options;

  options.SetOptimizationLevel(shaderc_optimization_level_performance);
  options.SetTargetSpirv(shaderc_spirv_version_1_6);
  options.SetTargetEnvironment(shaderc_target_env_vulkan,
                               shaderc_env_version_vulkan_1_3);

  shaderc::SpvCompilationResult module =
      compiler.CompileGlslToSpv(source, kind, "shader_src", options);

  if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
    std::cerr << module.GetErrorMessage() << std::endl;
    return VK_NULL_HANDLE;
  }

  std::vector<uint32_t> code{module.cbegin(), module.cend()};

  VkShaderModuleCreateInfo shaderInfo = {
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  shaderInfo.codeSize = code.size() * sizeof(code[0]);
  shaderInfo.pCode = code.data();
  VkShaderModule shader;
  vkCreateShaderModule(device, &shaderInfo, NULL, &shader);
  return shader;
}

}  // namespace

struct VxSorterLayout_T {
  VkDevice device = VK_NULL_HANDLE;

  VkDescriptorSetLayout storageDescriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout inoutDescriptorSetLayout = VK_NULL_HANDLE;

  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline histogramPipeline = VK_NULL_HANDLE;
  VkPipeline scanPipeline = VK_NULL_HANDLE;
  VkPipeline binningPipeline = VK_NULL_HANDLE;

  uint32_t histogramWorkgroupSize = 0;
};

struct VxSorter_T {
  VxSorterLayout layout = VK_NULL_HANDLE;
  VmaAllocator allocator = VK_NULL_HANDLE;

  uint32_t maxCommandsInFlight = 0;

  // indexing to descriptor, incremented after recording a command.
  uint32_t commandIndex = 0;

  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> storageDescriptors;
  std::vector<VkDescriptorSet> inOutDescriptors;
  std::vector<VkDescriptorSet> outInDescriptors;

  VkBuffer storage = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  StorageOffsets storageOffsets;
};

struct PushConstants {
  uint32_t elementCount;
  uint32_t pass;
};

void vxCreateSorterLayout(const VxSorterLayoutCreateInfo* pCreateInfo,
                          VxSorterLayout* pSorterLayout) {
  VkDevice device = pCreateInfo->device;

  // shader specialization constants and defaults
  uint32_t histogramWorkgroupSize = 1024;
  if (pCreateInfo->histogramWorkgroupSize != 0)
    histogramWorkgroupSize = pCreateInfo->histogramWorkgroupSize;

  // descriptor set layouts
  VkDescriptorSetLayout storageDescriptorSetLayout;
  {
    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings(2);
    descriptorSetLayoutBindings[0] = {};
    descriptorSetLayoutBindings[0].binding = 0;
    descriptorSetLayoutBindings[0].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBindings[0].descriptorCount = 1;
    descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptorSetLayoutBindings[1] = {};
    descriptorSetLayoutBindings[1].binding = 1;
    descriptorSetLayoutBindings[1].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBindings[1].descriptorCount = 1;
    descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptorSetLayoutInfo.bindingCount = descriptorSetLayoutBindings.size();
    descriptorSetLayoutInfo.pBindings = descriptorSetLayoutBindings.data();
    vkCreateDescriptorSetLayout(device, &descriptorSetLayoutInfo, NULL,
                                &storageDescriptorSetLayout);
  }

  VkDescriptorSetLayout inoutDescriptorSetLayout;
  {
    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings(2);
    descriptorSetLayoutBindings[0] = {};
    descriptorSetLayoutBindings[0].binding = 0;
    descriptorSetLayoutBindings[0].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBindings[0].descriptorCount = 1;
    descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptorSetLayoutBindings[1] = {};
    descriptorSetLayoutBindings[1].binding = 1;
    descriptorSetLayoutBindings[1].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBindings[1].descriptorCount = 1;
    descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptorSetLayoutInfo.bindingCount = descriptorSetLayoutBindings.size();
    descriptorSetLayoutInfo.pBindings = descriptorSetLayoutBindings.data();
    vkCreateDescriptorSetLayout(device, &descriptorSetLayoutInfo, NULL,
                                &inoutDescriptorSetLayout);
  }

  // pipeline layout
  VkPushConstantRange pushConstants = {};
  pushConstants.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushConstants.offset = 0;
  pushConstants.size = sizeof(PushConstants);

  std::vector<VkDescriptorSetLayout> setLayouts = {storageDescriptorSetLayout,
                                                   inoutDescriptorSetLayout};
  VkPipelineLayout pipelineLayout;
  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutInfo.setLayoutCount = setLayouts.size();
  pipelineLayoutInfo.pSetLayouts = setLayouts.data();
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges = &pushConstants;
  vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL, &pipelineLayout);

  // pipelines
  VkPipeline histogramPipeline;
  {
    VkShaderModule pipelineModule =
        CreateShaderModule(device, VK_SHADER_STAGE_COMPUTE_BIT, histogram_comp);

    VkSpecializationMapEntry mapEntry = {};
    mapEntry.constantID = 0;
    mapEntry.offset = 0;
    mapEntry.size = sizeof(histogramWorkgroupSize);

    VkSpecializationInfo specializationInfo = {};
    specializationInfo.mapEntryCount = 1;
    specializationInfo.pMapEntries = &mapEntry;
    specializationInfo.dataSize = sizeof(histogramWorkgroupSize);
    specializationInfo.pData = &histogramWorkgroupSize;

    VkComputePipelineCreateInfo pipelineInfo = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = pipelineModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.stage.pSpecializationInfo = &specializationInfo;
    pipelineInfo.layout = pipelineLayout;

    vkCreateComputePipelines(device, NULL, 1, &pipelineInfo, NULL,
                             &histogramPipeline);

    vkDestroyShaderModule(device, pipelineModule, NULL);
  }

  VkPipeline scanPipeline;
  {
    VkShaderModule pipelineModule =
        CreateShaderModule(device, VK_SHADER_STAGE_COMPUTE_BIT, scan_comp);

    VkComputePipelineCreateInfo pipelineInfo = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = pipelineModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;

    vkCreateComputePipelines(device, NULL, 1, &pipelineInfo, NULL,
                             &scanPipeline);

    vkDestroyShaderModule(device, pipelineModule, NULL);
  }

  VkPipeline binningPipeline;
  {
    VkShaderModule pipelineModule =
        CreateShaderModule(device, VK_SHADER_STAGE_COMPUTE_BIT, binning_comp);

    VkComputePipelineCreateInfo pipelineInfo = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = pipelineModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;

    vkCreateComputePipelines(device, NULL, 1, &pipelineInfo, NULL,
                             &binningPipeline);

    vkDestroyShaderModule(device, pipelineModule, NULL);
  }

  *pSorterLayout = new VxSorterLayout_T();
  (*pSorterLayout)->device = device;
  (*pSorterLayout)->storageDescriptorSetLayout = storageDescriptorSetLayout;
  (*pSorterLayout)->inoutDescriptorSetLayout = inoutDescriptorSetLayout;
  (*pSorterLayout)->pipelineLayout = pipelineLayout;
  (*pSorterLayout)->histogramPipeline = histogramPipeline;
  (*pSorterLayout)->scanPipeline = scanPipeline;
  (*pSorterLayout)->binningPipeline = binningPipeline;
  (*pSorterLayout)->histogramWorkgroupSize = histogramWorkgroupSize;
}

void vxDestroySorterLayout(VxSorterLayout sorterLayout) {
  vkDestroyPipeline(sorterLayout->device, sorterLayout->histogramPipeline,
                    NULL);
  vkDestroyPipeline(sorterLayout->device, sorterLayout->scanPipeline, NULL);
  vkDestroyPipeline(sorterLayout->device, sorterLayout->binningPipeline, NULL);
  vkDestroyPipelineLayout(sorterLayout->device, sorterLayout->pipelineLayout,
                          NULL);
  vkDestroyDescriptorSetLayout(sorterLayout->device,
                               sorterLayout->storageDescriptorSetLayout, NULL);
  vkDestroyDescriptorSetLayout(sorterLayout->device,
                               sorterLayout->inoutDescriptorSetLayout, NULL);
  delete sorterLayout;
}

void vxCreateSorter(const VxSorterCreateInfo* pCreateInfo, VxSorter* pSorter) {
  VxSorterLayout sorterLayout = pCreateInfo->sorterLayout;
  VkDevice device = sorterLayout->device;
  VmaAllocator allocator = pCreateInfo->allocator;
  uint32_t maxElementCount = pCreateInfo->maxElementCount;
  uint32_t frameCount = pCreateInfo->maxCommandsInFlight;

  // descriptor pool
  std::vector<VkDescriptorPoolSize> poolSizes = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * frameCount},
  };

  VkDescriptorPool descriptorPool;
  VkDescriptorPoolCreateInfo descriptorPoolInfo = {
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  descriptorPoolInfo.maxSets = 3 * frameCount;
  descriptorPoolInfo.poolSizeCount = poolSizes.size();
  descriptorPoolInfo.pPoolSizes = poolSizes.data();
  vkCreateDescriptorPool(device, &descriptorPoolInfo, NULL, &descriptorPool);

  // descriptors
  std::vector<VkDescriptorSet> storageDescriptors(frameCount);
  {
    std::vector<VkDescriptorSetLayout> setLayouts(
        frameCount, sorterLayout->storageDescriptorSetLayout);
    VkDescriptorSetAllocateInfo descriptorInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    descriptorInfo.descriptorPool = descriptorPool;
    descriptorInfo.descriptorSetCount = setLayouts.size();
    descriptorInfo.pSetLayouts = setLayouts.data();
    vkAllocateDescriptorSets(device, &descriptorInfo,
                             storageDescriptors.data());
  }

  std::vector<VkDescriptorSet> inOutDescriptors(frameCount);
  std::vector<VkDescriptorSet> outInDescriptors(frameCount);
  {
    std::vector<VkDescriptorSetLayout> setLayouts(
        frameCount, sorterLayout->inoutDescriptorSetLayout);
    VkDescriptorSetAllocateInfo descriptorInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    descriptorInfo.descriptorPool = descriptorPool;
    descriptorInfo.descriptorSetCount = setLayouts.size();
    descriptorInfo.pSetLayouts = setLayouts.data();
    vkAllocateDescriptorSets(device, &descriptorInfo, inOutDescriptors.data());
    vkAllocateDescriptorSets(device, &descriptorInfo, outInDescriptors.data());
  }

  // storage
  StorageOffsets storageOffsets;
  VkDeviceSize histogramSize = HistogramSize();
  VkDeviceSize lookbackSize = LookbackSize(maxElementCount);
  VkDeviceSize inoutSize = InoutSize(maxElementCount);
  storageOffsets.histogramOffset = 0;
  storageOffsets.lookbackOffset = Align(histogramSize, 64);
  storageOffsets.outOffset =
      Align(storageOffsets.lookbackOffset + lookbackSize, 64);
  VkDeviceSize storageSize = Align(storageOffsets.outOffset + inoutSize, 64);

  VkBuffer storage;
  VmaAllocation allocation;
  {
    VkBufferCreateInfo storageInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    storageInfo.size = storageSize;
    storageInfo.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    vmaCreateBuffer(allocator, &storageInfo, &allocationCreateInfo, &storage,
                    &allocation, NULL);
  }

  *pSorter = new VxSorter_T();
  (*pSorter)->layout = sorterLayout;
  (*pSorter)->allocator = allocator;
  (*pSorter)->maxCommandsInFlight = pCreateInfo->maxCommandsInFlight;
  (*pSorter)->descriptorPool = descriptorPool;
  (*pSorter)->storageDescriptors = std::move(storageDescriptors);
  (*pSorter)->inOutDescriptors = std::move(inOutDescriptors);
  (*pSorter)->outInDescriptors = std::move(outInDescriptors);
  (*pSorter)->storage = storage;
  (*pSorter)->allocation = allocation;
  (*pSorter)->storageOffsets = storageOffsets;
}

void vxDestroySorter(VxSorter sorter) {
  vkDestroyDescriptorPool(sorter->layout->device, sorter->descriptorPool, NULL);
  vmaDestroyBuffer(sorter->allocator, sorter->storage, sorter->allocation);
  delete sorter;
}

void vxCmdRadixSort(VkCommandBuffer commandBuffer, VxSorter sorter,
                    uint32_t elementCount, VkBuffer buffer, VkDeviceSize offset,
                    VkQueryPool queryPool, uint32_t query) {
  VxSorterLayout layout = sorter->layout;
  uint32_t commandIndex = sorter->commandIndex;
  VkBuffer storage = sorter->storage;
  VkDescriptorSet storageDescriptor = sorter->storageDescriptors[commandIndex];
  VkDescriptorSet inOutDescriptor = sorter->inOutDescriptors[commandIndex];
  VkDescriptorSet outInDescriptor = sorter->outInDescriptors[commandIndex];
  uint32_t partitionCount = RoundUp(elementCount, PARTITION_SIZE);

  constexpr VkDeviceSize histogramSize = HistogramSize();
  VkDeviceSize lookbackSize = LookbackSize(elementCount);
  VkDeviceSize inoutSize = InoutSize(elementCount);

  VkDeviceSize histogramOffset = sorter->storageOffsets.histogramOffset;
  VkDeviceSize lookbackOffset = sorter->storageOffsets.lookbackOffset;
  VkDeviceSize outOffset = sorter->storageOffsets.outOffset;

  // write descriptors
  std::vector<VkDescriptorBufferInfo> descriptorBuffers(4);
  descriptorBuffers[0].buffer = sorter->storage;
  descriptorBuffers[0].offset = histogramOffset;
  descriptorBuffers[0].range = histogramSize;

  descriptorBuffers[1].buffer = sorter->storage;
  descriptorBuffers[1].offset = lookbackOffset;
  descriptorBuffers[1].range = lookbackSize;

  descriptorBuffers[2].buffer = buffer;
  descriptorBuffers[2].offset = offset;
  descriptorBuffers[2].range = inoutSize;

  descriptorBuffers[3].buffer = sorter->storage;
  descriptorBuffers[3].offset = outOffset;
  descriptorBuffers[3].range = inoutSize;

  std::vector<VkWriteDescriptorSet> writes(6);
  writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  writes[0].dstSet = storageDescriptor;
  writes[0].dstBinding = 0;
  writes[0].dstArrayElement = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &descriptorBuffers[0];

  writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  writes[1].dstSet = storageDescriptor;
  writes[1].dstBinding = 1;
  writes[1].dstArrayElement = 0;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[1].pBufferInfo = &descriptorBuffers[1];

  writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  writes[2].dstSet = inOutDescriptor;
  writes[2].dstBinding = 0;
  writes[2].dstArrayElement = 0;
  writes[2].descriptorCount = 1;
  writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[2].pBufferInfo = &descriptorBuffers[2];

  writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  writes[3].dstSet = inOutDescriptor;
  writes[3].dstBinding = 1;
  writes[3].dstArrayElement = 0;
  writes[3].descriptorCount = 1;
  writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[3].pBufferInfo = &descriptorBuffers[3];

  writes[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  writes[4].dstSet = outInDescriptor;
  writes[4].dstBinding = 0;
  writes[4].dstArrayElement = 0;
  writes[4].descriptorCount = 1;
  writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[4].pBufferInfo = &descriptorBuffers[3];

  writes[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  writes[5].dstSet = outInDescriptor;
  writes[5].dstBinding = 1;
  writes[5].dstArrayElement = 0;
  writes[5].descriptorCount = 1;
  writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[5].pBufferInfo = &descriptorBuffers[2];

  vkUpdateDescriptorSets(layout->device, writes.size(), writes.data(), 0, NULL);

  if (queryPool) {
    vkCmdWriteTimestamp2(commandBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         queryPool, query + 0);
  }

  // clear histogram
  vkCmdFillBuffer(commandBuffer, storage, histogramOffset, histogramSize, 0);

  std::vector<VkBufferMemoryBarrier2> bufferMemoryBarriers(1);
  bufferMemoryBarriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
  bufferMemoryBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  bufferMemoryBarriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  bufferMemoryBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  bufferMemoryBarriers[0].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
  bufferMemoryBarriers[0].buffer = storage;
  bufferMemoryBarriers[0].offset = histogramOffset;
  bufferMemoryBarriers[0].size = histogramSize;
  VkDependencyInfo dependencyInfo = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependencyInfo.bufferMemoryBarrierCount = bufferMemoryBarriers.size();
  dependencyInfo.pBufferMemoryBarriers = bufferMemoryBarriers.data();
  vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

  // histogram
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    layout->histogramPipeline);

  PushConstants pushConstants;
  pushConstants.elementCount = elementCount;
  vkCmdPushConstants(commandBuffer, layout->pipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                     &pushConstants);

  std::vector<VkDescriptorSet> descriptors = {storageDescriptor,
                                              inOutDescriptor};
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          layout->pipelineLayout, 0, descriptors.size(),
                          descriptors.data(), 0, nullptr);

  vkCmdDispatch(commandBuffer,
                RoundUp(elementCount, layout->histogramWorkgroupSize), 1, 1);

  if (queryPool) {
    vkCmdWriteTimestamp2(commandBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         queryPool, query + 1);
  }

  // scan
  bufferMemoryBarriers.resize(1);
  bufferMemoryBarriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
  bufferMemoryBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  bufferMemoryBarriers[0].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
  bufferMemoryBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  bufferMemoryBarriers[0].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
  bufferMemoryBarriers[0].buffer = storage;
  bufferMemoryBarriers[0].offset = histogramOffset;
  bufferMemoryBarriers[0].size = histogramSize;
  dependencyInfo = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependencyInfo.bufferMemoryBarrierCount = bufferMemoryBarriers.size();
  dependencyInfo.pBufferMemoryBarriers = bufferMemoryBarriers.data();
  vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    layout->scanPipeline);

  vkCmdDispatch(commandBuffer, 1, 1, 1);

  if (queryPool) {
    vkCmdWriteTimestamp2(commandBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         queryPool, query + 2);
  }

  // binning passes
  bufferMemoryBarriers.resize(1);
  bufferMemoryBarriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
  bufferMemoryBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  bufferMemoryBarriers[0].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
  bufferMemoryBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  bufferMemoryBarriers[0].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
  bufferMemoryBarriers[0].buffer = storage;
  bufferMemoryBarriers[0].offset = histogramOffset;
  bufferMemoryBarriers[0].size = histogramSize;
  dependencyInfo = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependencyInfo.bufferMemoryBarrierCount = bufferMemoryBarriers.size();
  dependencyInfo.pBufferMemoryBarriers = bufferMemoryBarriers.data();
  vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    layout->binningPipeline);

  for (int i = 0; i < 4; i++) {
    // clear lookback buffer
    vkCmdFillBuffer(commandBuffer, storage, lookbackOffset, lookbackSize, 0);

    // binning
    bufferMemoryBarriers.resize(1);
    bufferMemoryBarriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    bufferMemoryBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    bufferMemoryBarriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    bufferMemoryBarriers[0].dstStageMask =
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    bufferMemoryBarriers[0].dstAccessMask =
        VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    bufferMemoryBarriers[0].buffer = storage;
    bufferMemoryBarriers[0].offset = lookbackOffset;
    bufferMemoryBarriers[0].size = lookbackSize;
    dependencyInfo = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependencyInfo.bufferMemoryBarrierCount = bufferMemoryBarriers.size();
    dependencyInfo.pBufferMemoryBarriers = bufferMemoryBarriers.data();
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

    VkDescriptorSet descriptor = i % 2 == 0 ? inOutDescriptor : outInDescriptor;
    std::vector<VkDescriptorSet> descriptors = {descriptor};
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            layout->pipelineLayout, 1, descriptors.size(),
                            descriptors.data(), 0, nullptr);

    PushConstants pushConstants;
    pushConstants.elementCount = elementCount;
    pushConstants.pass = i;
    vkCmdPushConstants(commandBuffer, layout->pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                       &pushConstants);

    vkCmdDispatch(commandBuffer, partitionCount, 1, 1);

    if (queryPool) {
      vkCmdWriteTimestamp2(commandBuffer,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, queryPool,
                           query + 3 + i);
    }

    if (i < 3) {
      bufferMemoryBarriers.resize(3);

      bufferMemoryBarriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
      bufferMemoryBarriers[0].srcStageMask =
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      bufferMemoryBarriers[0].srcAccessMask =
          VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
      bufferMemoryBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
      bufferMemoryBarriers[0].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      bufferMemoryBarriers[0].buffer = storage;
      bufferMemoryBarriers[0].offset = lookbackOffset;
      bufferMemoryBarriers[0].size = lookbackSize;

      bufferMemoryBarriers[1] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
      bufferMemoryBarriers[1].srcStageMask =
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      bufferMemoryBarriers[1].srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
      bufferMemoryBarriers[1].dstStageMask =
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      bufferMemoryBarriers[1].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
      bufferMemoryBarriers[1].buffer = i % 2 == 0 ? buffer : storage;
      bufferMemoryBarriers[1].offset = i % 2 == 0 ? offset : outOffset;
      bufferMemoryBarriers[1].size = inoutSize;

      bufferMemoryBarriers[2] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
      bufferMemoryBarriers[2].srcStageMask =
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      bufferMemoryBarriers[2].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
      bufferMemoryBarriers[2].dstStageMask =
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      bufferMemoryBarriers[2].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
      bufferMemoryBarriers[2].buffer = i % 2 == 0 ? storage : buffer;
      bufferMemoryBarriers[2].offset = i % 2 == 0 ? outOffset : offset;
      bufferMemoryBarriers[2].size = inoutSize;

      dependencyInfo = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      dependencyInfo.bufferMemoryBarrierCount = bufferMemoryBarriers.size();
      dependencyInfo.pBufferMemoryBarriers = bufferMemoryBarriers.data();
      vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
    }
  }

  if (queryPool) {
    vkCmdWriteTimestamp2(commandBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         queryPool, query + 7);
  }

  sorter->commandIndex = (commandIndex + 1) % sorter->maxCommandsInFlight;
}
