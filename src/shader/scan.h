#ifndef VK_RADIX_SORT_SHADER_SCAN_H
#define VK_RADIX_SORT_SHADER_SCAN_H

const char* scan_comp = R"shader(
#version 460 core

#extension GL_KHR_shader_subgroup_basic: enable
#extension GL_KHR_shader_subgroup_arithmetic: enable
#extension GL_KHR_shader_subgroup_ballot: enable

const uint RADIX = 256;

// dispatch with group count (1, 1, 1)
layout (local_size_x = RADIX) in;

layout (set = 0, binding = 0) buffer Histogram {
  uint histogram[];  // (4, R)
};

shared uint scanIntermediate[RADIX / 32];

void main() {
  uint threadIndex = gl_SubgroupInvocationID;  // 0..31
  uint subgroupIndex = gl_SubgroupID;  // 0..7
  uint index = subgroupIndex * gl_SubgroupSize + threadIndex;

  uint excl[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint value = histogram[RADIX * i + index];
    excl[i] = subgroupExclusiveAdd(value);
    uint sum = subgroupAdd(value);
    if (threadIndex == 0) {
      scanIntermediate[gl_NumSubgroups * i + subgroupIndex] = sum;
    }
  }
  barrier();

  if (index < RADIX / gl_SubgroupSize) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint value = scanIntermediate[gl_NumSubgroups * i + index];
      uint excl = subgroupExclusiveAdd(value);
      scanIntermediate[gl_NumSubgroups * i + index] = excl;
    }
  }
  barrier();

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint broadcastValue;
    if (threadIndex == 0) {
      broadcastValue = scanIntermediate[gl_NumSubgroups * i + subgroupIndex];
    }
    uint scanSum = subgroupBroadcast(broadcastValue, 0);
    histogram[RADIX * i + index] = scanSum + excl[i];
  }
}

)shader";

#endif  // VK_RADIX_SORT_SHADER_SCAN_H
