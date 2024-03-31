#include <vk_radix_sort.h>

#include <iostream>
#include <vector>
#include <string>

#include <shaderc/shaderc.hpp>

#include "shader/histogram.h"

namespace {

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

  VkShaderModuleCreateInfo shader_info = {
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  shader_info.codeSize = code.size() * sizeof(code[0]);
  shader_info.pCode = code.data();
  VkShaderModule shader;
  vkCreateShaderModule(device, &shader_info, NULL, &shader);
  return shader;
}

}  // namespace

struct VxSorter_T {
  VkDevice device = VK_NULL_HANDLE;

  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
  VkDescriptorSet descriptor = VK_NULL_HANDLE;

  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkPipeline histogram_pipeline = VK_NULL_HANDLE;
};

struct PushConstants {
  uint32_t num_elements;
};

void vxCreateSorter(const VxSorterCreateInfo* pCreateInfo, VxSorter* pSorter) {
  VkDevice device = pCreateInfo->device;

  std::vector<VkDescriptorPoolSize> pool_sizes = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16},
  };

  // descriptor pool
  VkDescriptorPool descriptor_pool;
  VkDescriptorPoolCreateInfo descriptor_pool_info = {
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  descriptor_pool_info.maxSets = 16;
  descriptor_pool_info.poolSizeCount = pool_sizes.size();
  descriptor_pool_info.pPoolSizes = pool_sizes.data();
  vkCreateDescriptorPool(device, &descriptor_pool_info, NULL, &descriptor_pool);

  // descriptor set layout
  std::vector<VkDescriptorSetLayoutBinding> descriptor_set_layout_bindings(3);
  descriptor_set_layout_bindings[0] = {};
  descriptor_set_layout_bindings[0].binding = 0;
  descriptor_set_layout_bindings[0].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptor_set_layout_bindings[0].descriptorCount = 1;
  descriptor_set_layout_bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  descriptor_set_layout_bindings[1] = {};
  descriptor_set_layout_bindings[1].binding = 1;
  descriptor_set_layout_bindings[1].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptor_set_layout_bindings[1].descriptorCount = 1;
  descriptor_set_layout_bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  descriptor_set_layout_bindings[2] = {};
  descriptor_set_layout_bindings[2].binding = 2;
  descriptor_set_layout_bindings[2].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptor_set_layout_bindings[2].descriptorCount = 1;
  descriptor_set_layout_bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayout descriptor_set_layout;
  VkDescriptorSetLayoutCreateInfo descriptor_set_layout_info = {
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  descriptor_set_layout_info.bindingCount =
      descriptor_set_layout_bindings.size();
  descriptor_set_layout_info.pBindings = descriptor_set_layout_bindings.data();
  vkCreateDescriptorSetLayout(device, &descriptor_set_layout_info, NULL,
                              &descriptor_set_layout);

  // pipeline layout
  VkPushConstantRange push_constants = {};
  push_constants.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_constants.offset = 0;
  push_constants.size = sizeof(PushConstants);

  VkPipelineLayout pipeline_layout;
  VkPipelineLayoutCreateInfo pipeline_layout_info = {
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &descriptor_set_layout;
  pipeline_layout_info.pushConstantRangeCount = 1;
  pipeline_layout_info.pPushConstantRanges = &push_constants;
  vkCreatePipelineLayout(device, &pipeline_layout_info, NULL, &pipeline_layout);

  // pipelines
  VkShaderModule histogram_pipeline_module =
      CreateShaderModule(device, VK_SHADER_STAGE_COMPUTE_BIT, histogram_comp);

  VkPipeline histogram_pipeline;
  VkComputePipelineCreateInfo histogram_pipeline_info = {
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  histogram_pipeline_info.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  histogram_pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  histogram_pipeline_info.stage.module = histogram_pipeline_module;
  histogram_pipeline_info.stage.pName = "main";
  histogram_pipeline_info.layout = pipeline_layout;

  vkCreateComputePipelines(device, NULL, 1, &histogram_pipeline_info, NULL,
                           &histogram_pipeline);

  vkDestroyShaderModule(device, histogram_pipeline_module, NULL);

  *pSorter = new VxSorter_T();
  (*pSorter)->device = device;
  (*pSorter)->descriptor_pool = descriptor_pool;
  (*pSorter)->descriptor_set_layout = descriptor_set_layout;
  (*pSorter)->pipeline_layout = pipeline_layout;
  (*pSorter)->histogram_pipeline = histogram_pipeline;
}

void vxDestroySorter(VxSorter sorter) {
  vkDestroyPipeline(sorter->device, sorter->histogram_pipeline, NULL);
  vkDestroyPipelineLayout(sorter->device, sorter->pipeline_layout, NULL);
  vkDestroyDescriptorSetLayout(sorter->device, sorter->descriptor_set_layout,
                               NULL);
  vkDestroyDescriptorPool(sorter->device, sorter->descriptor_pool, NULL);
  delete sorter;
}

void vxCmdRadixSort(VkCommandBuffer commandBuffer, VxSorter sorter,
                    uint32_t elementCount) {
  vxCmdRadixSortGlobalHistogram(commandBuffer, sorter, elementCount);

  // TODO: scan
  // TODO: binning 0..3
}

void vxCmdRadixSortGlobalHistogram(VkCommandBuffer commandBuffer,
                                   VxSorter sorter, uint32_t elementCount) {
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    sorter->histogram_pipeline);

  PushConstants push_constants;
  push_constants.num_elements = elementCount;
  vkCmdPushConstants(commandBuffer, sorter->pipeline_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants),
                     &push_constants);

  // TODO: bind descriptor set
  // TODO: dispatch
}
