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
constexpr uint32_t WORKGROUP_SIZE = 512;

uint32_t RoundUp(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

constexpr VkDeviceSize HistogramByteSize() {
  return 4 * RADIX * sizeof(uint32_t);
}

VkDeviceSize InoutByteSize(uint32_t elementCount) {
  return elementCount * sizeof(uint32_t);
}

VkDeviceSize LookbackByteSize(uint32_t elementCount) {
  return (1 + RADIX * RoundUp(elementCount, PARTITION_SIZE)) * sizeof(uint32_t);
}

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

struct VxSorter_T {
  VkDevice device = VK_NULL_HANDLE;
  uint32_t maxCommandsInFlight = 0;

  // indexing to descriptor, incremented after recording a command.
  uint32_t commandIndex = 0;

  VkDescriptorSetLayout storageDescriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout inoutDescriptorSetLayout = VK_NULL_HANDLE;

  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline histogramPipeline = VK_NULL_HANDLE;
  VkPipeline scanPipeline = VK_NULL_HANDLE;
  VkPipeline binningPipeline = VK_NULL_HANDLE;

  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> storageDescriptors;
  std::vector<VkDescriptorSet> inoutDescriptors;
};

struct PushConstants {
  uint32_t elementCount;
  uint32_t pass;
};

void vxCreateSorter(const VxSorterCreateInfo* pCreateInfo, VxSorter* pSorter) {
  VkDevice device = pCreateInfo->device;
  uint32_t frameCount = pCreateInfo->maxCommandsInFlight;

  // descriptor set layouts
  VkDescriptorSetLayout storageDescriptorSetLayout;
  {
    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings(3);
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

    descriptorSetLayoutBindings[2] = {};
    descriptorSetLayoutBindings[2].binding = 2;
    descriptorSetLayoutBindings[2].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBindings[2].descriptorCount = 1;
    descriptorSetLayoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

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

    VkComputePipelineCreateInfo pipelineInfo = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = pipelineModule;
    pipelineInfo.stage.pName = "main";
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

  // descriptor pool
  std::vector<VkDescriptorPoolSize> poolSizes = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * frameCount},
  };

  VkDescriptorPool descriptorPool;
  VkDescriptorPoolCreateInfo descriptorPoolInfo = {
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  descriptorPoolInfo.maxSets = 2 * frameCount;
  descriptorPoolInfo.poolSizeCount = poolSizes.size();
  descriptorPoolInfo.pPoolSizes = poolSizes.data();
  vkCreateDescriptorPool(device, &descriptorPoolInfo, NULL, &descriptorPool);

  // descriptors
  std::vector<VkDescriptorSet> storageDescriptors(frameCount);
  {
    std::vector<VkDescriptorSetLayout> setLayouts(frameCount,
                                                  storageDescriptorSetLayout);
    VkDescriptorSetAllocateInfo descriptorInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    descriptorInfo.descriptorPool = descriptorPool;
    descriptorInfo.descriptorSetCount = setLayouts.size();
    descriptorInfo.pSetLayouts = setLayouts.data();
    vkAllocateDescriptorSets(device, &descriptorInfo,
                             storageDescriptors.data());
  }

  std::vector<VkDescriptorSet> inoutDescriptors(frameCount);
  {
    std::vector<VkDescriptorSetLayout> setLayouts(frameCount,
                                                  inoutDescriptorSetLayout);
    VkDescriptorSetAllocateInfo descriptorInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    descriptorInfo.descriptorPool = descriptorPool;
    descriptorInfo.descriptorSetCount = setLayouts.size();
    descriptorInfo.pSetLayouts = setLayouts.data();
    vkAllocateDescriptorSets(device, &descriptorInfo, inoutDescriptors.data());
  }

  *pSorter = new VxSorter_T();
  (*pSorter)->device = device;
  (*pSorter)->maxCommandsInFlight = pCreateInfo->maxCommandsInFlight;
  (*pSorter)->descriptorPool = descriptorPool;
  (*pSorter)->storageDescriptorSetLayout = storageDescriptorSetLayout;
  (*pSorter)->inoutDescriptorSetLayout = inoutDescriptorSetLayout;
  (*pSorter)->pipelineLayout = pipelineLayout;
  (*pSorter)->histogramPipeline = histogramPipeline;
  (*pSorter)->scanPipeline = scanPipeline;
  (*pSorter)->binningPipeline = binningPipeline;
  (*pSorter)->storageDescriptors = std::move(storageDescriptors);
  (*pSorter)->inoutDescriptors = std::move(inoutDescriptors);
}

void vxDestroySorter(VxSorter sorter) {
  vkDestroyPipeline(sorter->device, sorter->histogramPipeline, NULL);
  vkDestroyPipeline(sorter->device, sorter->scanPipeline, NULL);
  vkDestroyPipeline(sorter->device, sorter->binningPipeline, NULL);
  vkDestroyPipelineLayout(sorter->device, sorter->pipelineLayout, NULL);
  vkDestroyDescriptorSetLayout(sorter->device,
                               sorter->storageDescriptorSetLayout, NULL);
  vkDestroyDescriptorSetLayout(sorter->device, sorter->inoutDescriptorSetLayout,
                               NULL);
  vkDestroyDescriptorPool(sorter->device, sorter->descriptorPool, NULL);
  delete sorter;
}

void vxCmdRadixSort(VkCommandBuffer commandBuffer, VxSorter sorter,
                    uint32_t elementCount) {
  // TODO: global histogram
  // TODO: scan
  // TODO: binning 0..3
}

void vxCmdRadixSortGlobalHistogram(VkCommandBuffer commandBuffer,
                                   VxSorter sorter, uint32_t elementCount,
                                   VkBuffer buffer, VkDeviceSize offset,
                                   VkBuffer histogramBuffer,
                                   VkDeviceSize histogramOffset) {
  uint32_t commandIndex = sorter->commandIndex;
  VkDescriptorSet storageDescriptor = sorter->storageDescriptors[commandIndex];
  VkDescriptorSet inoutDescriptor = sorter->inoutDescriptors[commandIndex];
  VkDeviceSize inoutBufferSize = InoutByteSize(elementCount);

  // write descriptors
  std::vector<VkDescriptorBufferInfo> descriptorBuffers(2);
  descriptorBuffers[0].buffer = histogramBuffer;
  descriptorBuffers[0].offset = histogramOffset;
  descriptorBuffers[0].range = HistogramByteSize();

  descriptorBuffers[1].buffer = buffer;
  descriptorBuffers[1].offset = offset;
  descriptorBuffers[1].range = inoutBufferSize;

  std::vector<VkWriteDescriptorSet> writes(2);
  writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  writes[0].dstSet = storageDescriptor;
  writes[0].dstBinding = 0;
  writes[0].dstArrayElement = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &descriptorBuffers[0];

  writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  writes[1].dstSet = inoutDescriptor;
  writes[1].dstBinding = 0;
  writes[1].dstArrayElement = 0;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[1].pBufferInfo = &descriptorBuffers[1];

  vkUpdateDescriptorSets(sorter->device, writes.size(), writes.data(), 0, NULL);

  // fill command
  vkCmdFillBuffer(commandBuffer, histogramBuffer, histogramOffset,
                  HistogramByteSize(), 0);

  std::vector<VkBufferMemoryBarrier2> bufferMemoryBarriers(1);
  bufferMemoryBarriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
  bufferMemoryBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  bufferMemoryBarriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  bufferMemoryBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  bufferMemoryBarriers[0].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
  bufferMemoryBarriers[0].buffer = histogramBuffer;
  bufferMemoryBarriers[0].offset = 0;
  bufferMemoryBarriers[0].size = HistogramByteSize();
  VkDependencyInfo dependencyInfo = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependencyInfo.bufferMemoryBarrierCount = bufferMemoryBarriers.size();
  dependencyInfo.pBufferMemoryBarriers = bufferMemoryBarriers.data();
  vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

  // histogram
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    sorter->histogramPipeline);

  PushConstants pushConstants;
  pushConstants.elementCount = elementCount;
  vkCmdPushConstants(commandBuffer, sorter->pipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                     &pushConstants);

  std::vector<VkDescriptorSet> descriptors = {storageDescriptor,
                                              inoutDescriptor};
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          sorter->pipelineLayout, 0, descriptors.size(),
                          descriptors.data(), 0, nullptr);

  vkCmdDispatch(commandBuffer, RoundUp(elementCount, WORKGROUP_SIZE), 1, 1);

  sorter->commandIndex = (commandIndex + 1) % sorter->maxCommandsInFlight;
}

void vxCmdRadixSortGlobalHistogramScan(VkCommandBuffer commandBuffer,
                                       VxSorter sorter,
                                       VkBuffer histogramBuffer,
                                       VkDeviceSize histogramOffset,
                                       VkBuffer scanBuffer,
                                       VkDeviceSize scanOffset) {
  uint32_t commandIndex = sorter->commandIndex;
  VkDescriptorSet storageDescriptor = sorter->storageDescriptors[commandIndex];

  // write descriptors
  std::vector<VkDescriptorBufferInfo> descriptorBuffers(2);
  descriptorBuffers[0].buffer = histogramBuffer;
  descriptorBuffers[0].offset = histogramOffset;
  descriptorBuffers[0].range = HistogramByteSize();

  descriptorBuffers[1].buffer = scanBuffer;
  descriptorBuffers[1].offset = scanOffset;
  descriptorBuffers[1].range = HistogramByteSize();

  std::vector<VkWriteDescriptorSet> writes(2);
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

  vkUpdateDescriptorSets(sorter->device, writes.size(), writes.data(), 0, NULL);

  // scan
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    sorter->scanPipeline);

  std::vector<VkDescriptorSet> descriptors = {storageDescriptor};
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          sorter->pipelineLayout, 0, descriptors.size(),
                          descriptors.data(), 0, nullptr);

  vkCmdDispatch(commandBuffer, 1, 1, 1);

  sorter->commandIndex = (commandIndex + 1) % sorter->maxCommandsInFlight;
}

void vxCmdRadixSortBinning(VkCommandBuffer commandBuffer, VxSorter sorter,
                           uint32_t elementCount, uint32_t pass,
                           VkBuffer buffer, VkDeviceSize offset,
                           VkBuffer scanBuffer, VkDeviceSize scanOffset,
                           VkBuffer lookbackBuffer, VkDeviceSize lookbackOffset,
                           VkBuffer outBuffer, VkDeviceSize outOffset) {
  uint32_t commandIndex = sorter->commandIndex;
  VkDescriptorSet storageDescriptor = sorter->storageDescriptors[commandIndex];
  VkDescriptorSet inoutDescriptor = sorter->inoutDescriptors[commandIndex];
  VkDeviceSize inoutBufferSize = InoutByteSize(elementCount);
  VkDeviceSize lookbackBufferSize = LookbackByteSize(elementCount);
  uint32_t partitionCount = RoundUp(elementCount, PARTITION_SIZE);

  // write descriptors
  std::vector<VkDescriptorBufferInfo> descriptorBuffers(4);
  descriptorBuffers[0].buffer = scanBuffer;
  descriptorBuffers[0].offset = scanOffset;
  descriptorBuffers[0].range = HistogramByteSize();

  descriptorBuffers[1].buffer = lookbackBuffer;
  descriptorBuffers[1].offset = lookbackOffset;
  descriptorBuffers[1].range = lookbackBufferSize;

  descriptorBuffers[2].buffer = buffer;
  descriptorBuffers[2].offset = offset;
  descriptorBuffers[2].range = inoutBufferSize;

  descriptorBuffers[3].buffer = outBuffer;
  descriptorBuffers[3].offset = outOffset;
  descriptorBuffers[3].range = inoutBufferSize;

  std::vector<VkWriteDescriptorSet> writes(4);
  writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  writes[0].dstSet = storageDescriptor;
  writes[0].dstBinding = 1;
  writes[0].dstArrayElement = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &descriptorBuffers[0];

  writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  writes[1].dstSet = storageDescriptor;
  writes[1].dstBinding = 2;
  writes[1].dstArrayElement = 0;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[1].pBufferInfo = &descriptorBuffers[1];

  writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  writes[2].dstSet = inoutDescriptor;
  writes[2].dstBinding = 0;
  writes[2].dstArrayElement = 0;
  writes[2].descriptorCount = 1;
  writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[2].pBufferInfo = &descriptorBuffers[2];

  writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  writes[3].dstSet = inoutDescriptor;
  writes[3].dstBinding = 1;
  writes[3].dstArrayElement = 0;
  writes[3].descriptorCount = 1;
  writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[3].pBufferInfo = &descriptorBuffers[3];

  vkUpdateDescriptorSets(sorter->device, writes.size(), writes.data(), 0, NULL);

  // clear lookback buffer
  vkCmdFillBuffer(commandBuffer, lookbackBuffer, 0, lookbackBufferSize, 0);

  // binning
  std::vector<VkBufferMemoryBarrier2> bufferMemoryBarriers(1);
  bufferMemoryBarriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
  bufferMemoryBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  bufferMemoryBarriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  bufferMemoryBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  bufferMemoryBarriers[0].dstAccessMask =
      VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
  bufferMemoryBarriers[0].buffer = lookbackBuffer;
  bufferMemoryBarriers[0].offset = 0;
  bufferMemoryBarriers[0].size = lookbackBufferSize;
  VkDependencyInfo dependencyInfo = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependencyInfo.bufferMemoryBarrierCount = bufferMemoryBarriers.size();
  dependencyInfo.pBufferMemoryBarriers = bufferMemoryBarriers.data();
  vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    sorter->binningPipeline);

  std::vector<VkDescriptorSet> descriptors = {storageDescriptor,
                                              inoutDescriptor};
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          sorter->pipelineLayout, 0, descriptors.size(),
                          descriptors.data(), 0, nullptr);

  PushConstants pushConstants;
  pushConstants.elementCount = elementCount;
  pushConstants.pass = pass;
  vkCmdPushConstants(commandBuffer, sorter->pipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                     &pushConstants);

  vkCmdDispatch(commandBuffer, partitionCount, 1, 1);

  sorter->commandIndex = (commandIndex + 1) % sorter->maxCommandsInFlight;
}
