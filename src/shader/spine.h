#ifndef VK_RADIX_SORT_SHADER_SPINE_H
#define VK_RADIX_SORT_SHADER_SPINE_H

const char* spine_comp = R"shader(
#version 460 core

#extension GL_KHR_shader_subgroup_basic: enable
#extension GL_KHR_shader_subgroup_arithmetic: enable
#extension GL_KHR_shader_subgroup_ballot: enable

const int RADIX = 256;
// #define SUBGROUP_SIZE 32
// #define WORKGROUP_SIZE 512
// #define PARTITION_DIVISION 8
const int PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;

// dispatch this shader (RADIX, 1, 1), so that gl_WorkGroupID.x is radix
layout (local_size_x = WORKGROUP_SIZE) in;

layout (push_constant) uniform PushConstant {
  int pass;
};

layout (set = 0, binding = 1, std430) buffer Histogram {
  uint globalHistogram[4 * RADIX];  // (4, R)
  uint partitionHistogram[];  // (P, R)
};

layout (set = 0, binding = 2) readonly buffer ElementCount {
  uint elementCount;
};

shared uint reduction;
shared uint intermediate[SUBGROUP_SIZE];

void main() {
  uint threadIndex = gl_SubgroupInvocationID;  // 0..31
  uint subgroupIndex = gl_SubgroupID;  // 0..15
  uint index = subgroupIndex * gl_SubgroupSize + threadIndex;
  uint radix = gl_WorkGroupID.x;

  uint partitionCount = (elementCount + PARTITION_SIZE - 1) / PARTITION_SIZE;

  if (index == 0) {
    reduction = 0;
  }
  barrier();

  for (uint i = 0; WORKGROUP_SIZE * i < partitionCount; ++i) {
    uint partitionIndex = WORKGROUP_SIZE * i + index;
    uint value = partitionIndex < partitionCount ? partitionHistogram[RADIX * partitionIndex + radix] : 0;
    uint excl = subgroupExclusiveAdd(value) + reduction;
    uint sum = subgroupAdd(value);

    if (subgroupElect()) {
      intermediate[subgroupIndex] = sum;
    }
    barrier();

    if (index < gl_NumSubgroups) {
      uint excl = subgroupExclusiveAdd(intermediate[index]);
      uint sum = subgroupAdd(intermediate[index]);
      intermediate[index] = excl;

      if (index == 0) {
        reduction += sum;
      }
    }
    barrier();

    if (partitionIndex < partitionCount) {
      excl += intermediate[subgroupIndex];
      partitionHistogram[RADIX * partitionIndex + radix] = excl;
    }
    barrier();
  }

  if (gl_WorkGroupID.x == 0) {
    // one workgroup is responsible for global histogram prefix sum
    if (index < RADIX) {
      uint value = globalHistogram[RADIX * pass + index];
      uint excl = subgroupExclusiveAdd(value);
      uint sum = subgroupAdd(value);

      if (subgroupElect()) {
        intermediate[subgroupIndex] = sum;
      }
      barrier();

      if (index < RADIX / gl_SubgroupSize) {
        uint excl = subgroupExclusiveAdd(intermediate[index]);
        intermediate[index] = excl;
      }
      barrier();

      excl += intermediate[subgroupIndex];
      globalHistogram[RADIX * pass + index] = excl;
    }
  }
}

)shader";

#endif  // VK_RADIX_SORT_SHADER_SPINE_H