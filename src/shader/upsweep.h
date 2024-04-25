#ifndef VK_RADIX_SORT_SHADER_UPSWEEP_H
#define VK_RADIX_SORT_SHADER_UPSWEEP_H

const char* upsweep_comp = R"shader(
#version 460 core

#extension GL_KHR_shader_subgroup_basic: enable

const int RADIX = 256;
// #define WORKGROUP_SIZE 512
// #define PARTITION_DIVISION 8
const int PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;

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

layout (set = 1, binding = 0) readonly buffer Keys {
  uint keys[];  // (N)
};

shared uint localHistogram[RADIX];

void main() {
  uint threadIndex = gl_SubgroupInvocationID;  // 0..31
  uint subgroupIndex = gl_SubgroupID;  // 0..31
  uint index = subgroupIndex * gl_SubgroupSize + threadIndex;

  uint partitionIndex = gl_WorkGroupID.x;
  uint partitionStart = partitionIndex * PARTITION_SIZE;

  // discard all workgroup invocations
  if (partitionStart >= elementCount) {
    return;
  }

  if (index < RADIX) {
    localHistogram[index] = 0;
  }
  barrier();

  // local histogram
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint keyIndex = partitionStart + WORKGROUP_SIZE * i + index;
    uint key = keyIndex < elementCount ? keys[keyIndex] : 0xffffffff;
    uint radix = bitfieldExtract(key, 8 * pass, 8);
    atomicAdd(localHistogram[radix], 1);
  }
  barrier();

  if (index < RADIX) {
    // set to partition histogram
    partitionHistogram[RADIX * partitionIndex + index] = localHistogram[index];

    // add to global histogram
    atomicAdd(globalHistogram[RADIX * pass + index], localHistogram[index]);
  }
}

)shader";

#endif  // VK_RADIX_SORT_SHADER_UPSWEEP_H
