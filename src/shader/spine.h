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

shared uint intermediate[SUBGROUP_SIZE];

void main() {
  uint threadIndex = gl_SubgroupInvocationID;  // 0..31
  uint subgroupIndex = gl_SubgroupID;  // 0..31
  uint index = subgroupIndex * gl_SubgroupSize + threadIndex;
  uint radix = gl_WorkGroupID.x;

  uint partitionCount = (elementCount + PARTITION_SIZE - 1) / PARTITION_SIZE;

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

  for (uint i = index; i < partitionCount; i += WORKGROUP_SIZE) {
    // only for active invocations, index consecutive from 0..j. good for subgroup operations below.
    uint value = partitionHistogram[RADIX * i + radix];
    uint excl = subgroupExclusiveAdd(value);
    uint sum = subgroupAdd(value);

    if (subgroupElect()) {
      intermediate[subgroupIndex] = sum;
    }
    barrier();

    if (index < gl_SubgroupSize) {
      uint excl = subgroupExclusiveAdd(intermediate[index]);
      intermediate[index] = excl;
    }
    barrier();

    excl += intermediate[subgroupIndex];
    partitionHistogram[RADIX * i + radix] = sum;
    barrier();
  }
}

)shader";

#endif  // VK_RADIX_SORT_SHADER_SPINE_H
