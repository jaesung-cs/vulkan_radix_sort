#ifndef VK_RADIX_SORT_SHADER_DOWNSWEEP_H
#define VK_RADIX_SORT_SHADER_DOWNSWEEP_H

const char* downsweep_comp = R"shader(
#version 460 core

#extension GL_KHR_shader_subgroup_basic: enable
#extension GL_KHR_shader_subgroup_arithmetic: enable
#extension GL_KHR_shader_subgroup_ballot: enable

const int RADIX = 256;
// #define WORKGROUP_SIZE 512
// #define PARTITION_DIVISION 8
const int PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;

layout (local_size_x = WORKGROUP_SIZE) in;

layout (push_constant) uniform PushConstant {
  int pass;
};

layout (set = 0, binding = 1, std430) readonly buffer Histogram {
  uint globalHistogram[4 * RADIX];  // (4, R)
  uint partitionHistogram[];  // (P, R)
};

layout (set = 0, binding = 2) readonly buffer ElementCount {
  uint elementCount;
};

layout (set = 1, binding = 0) readonly buffer Keys {
  uint keys[];  // (N)
};

layout (set = 1, binding = 1) writeonly buffer OutKeys {
  uint outKeys[];  // (N)
};

#ifdef KEY_VALUE
layout (set = 1, binding = 2) readonly buffer Values {
  uint values[];  // (N)
};

layout (set = 1, binding = 3) writeonly buffer OutValues {
  uint outValues[];  // (N)
};

const uint SHMEM_SIZE = 2 * PARTITION_SIZE;
#else
const uint SHMEM_SIZE = PARTITION_SIZE;
#endif

shared uint localHistogram[SHMEM_SIZE];  // (R, S=16)=4096, (P), or (2P) for alias. take maximum.
shared uint localHistogramSum[RADIX];

// returns 0b00000....11111, where msb is id-1.
uvec4 GetExclusiveSubgroupMask(uint id) {
  return uvec4(
    (1 << id) - 1,
    (1 << (id - 32)) - 1,
    (1 << (id - 64)) - 1,
    (1 << (id - 96)) - 1
  );
}

uint GetBitCount(uvec4 value) {
  uvec4 result = bitCount(value);
  return result[0] + result[1] + result[2] + result[3];
}

void main() {
  uint threadIndex = gl_SubgroupInvocationID;  // 0..31
  uint subgroupIndex = gl_SubgroupID;  // 0..15
  uint index = subgroupIndex * gl_SubgroupSize + threadIndex;
  uvec4 subgroupMask = GetExclusiveSubgroupMask(threadIndex);

  uint partitionIndex = gl_WorkGroupID.x;
  uint partitionStart = partitionIndex * PARTITION_SIZE;

  if (index < RADIX) {
    for (int i = 0; i < gl_NumSubgroups; ++i) {
      localHistogram[gl_NumSubgroups * index + i] = 0;
    }
  }
  barrier();

  // load from global memory, local histogram and offset
  uint localKeys[PARTITION_DIVISION];
  uint localRadix[PARTITION_DIVISION];
  uint localOffsets[PARTITION_DIVISION];
  uint subgroupHistogram[PARTITION_DIVISION];

#ifdef KEY_VALUE
  uint localValues[PARTITION_DIVISION];
#endif
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint keyIndex = partitionStart + (PARTITION_DIVISION * gl_SubgroupSize) * subgroupIndex + i * gl_SubgroupSize + threadIndex;
    uint key = keyIndex < elementCount ? keys[keyIndex] : 0xffffffff;
    localKeys[i] = key;
    
#ifdef KEY_VALUE
    localValues[i] = keyIndex < elementCount ? values[keyIndex] : 0;
#endif

    uint radix = bitfieldExtract(key, pass * 8, 8);
    localRadix[i] = radix;

    // mask per digit
    uvec4 mask = subgroupBallot(true);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint digit = (radix >> j) & 1;
      uvec4 ballot = subgroupBallot(digit == 1);
      // digit - 1 is 0 or 0xffffffff. xor to flip.
      mask &= uvec4(digit - 1) ^ ballot;
    }

    // subgroup level offset for radix
    uint subgroupOffset = GetBitCount(subgroupMask & mask);
    uint radixCount = GetBitCount(mask);

    // elect a representative per radix, add to histogram
    if (subgroupOffset == 0) {
      // accumulate to local histogram
      atomicAdd(localHistogram[gl_NumSubgroups * radix + subgroupIndex], radixCount);
      subgroupHistogram[i] = radixCount;
    } else {
      subgroupHistogram[i] = 0;
    }

    localOffsets[i] = subgroupOffset;
  }
  
  // local histogram reduce 4096
  for (uint i = index; i < RADIX * gl_NumSubgroups; i += WORKGROUP_SIZE) {
    uint v = localHistogram[i];
    uint sum = subgroupAdd(v);
    uint excl = subgroupExclusiveAdd(v);
    localHistogram[i] = excl;
    if (threadIndex == 0) {
      localHistogramSum[i / gl_SubgroupSize] = sum;
    }
  }
  barrier();

  // local histogram reduce 128
  uint intermediateOffset0 = RADIX * gl_NumSubgroups / gl_SubgroupSize;
  if (index < intermediateOffset0) {
    uint v = localHistogramSum[index];
    uint sum = subgroupAdd(v);
    uint excl = subgroupExclusiveAdd(v);
    localHistogramSum[index] = excl;
    if (threadIndex == 0) {
      localHistogramSum[intermediateOffset0 + index / gl_SubgroupSize] = sum;
    }
  }
  barrier();

  // local histogram reduce 4
  uint intermediateSize1 = RADIX * gl_NumSubgroups / gl_SubgroupSize / gl_SubgroupSize;
  if (index < intermediateSize1) {
    uint v = localHistogramSum[intermediateOffset0 + index];
    uint excl = subgroupExclusiveAdd(v);
    localHistogramSum[intermediateOffset0 + index] = excl;
  }
  barrier();

  // local histogram add 128
  if (index < intermediateOffset0) {
    localHistogramSum[index] += localHistogramSum[intermediateOffset0 + index / gl_SubgroupSize];
  }
  barrier();

  // local histogram add 4096
  for (uint i = index; i < RADIX * gl_NumSubgroups; i += WORKGROUP_SIZE) {
    localHistogram[i] += localHistogramSum[i / gl_SubgroupSize];
  }
  barrier();

  // post-scan stage
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint radix = localRadix[i];
    localOffsets[i] += localHistogram[gl_NumSubgroups * radix + subgroupIndex];

    barrier();
    if (subgroupHistogram[i] > 0) {
      atomicAdd(localHistogram[gl_NumSubgroups * radix + subgroupIndex], subgroupHistogram[i]);
    }
    barrier();
  }

  // after atomicAdd, localHistogram contains inclusive sum
  if (index < RADIX) {
    uint v = localHistogram[gl_NumSubgroups * (index + 1) - 1];
    localHistogramSum[index] = globalHistogram[RADIX * pass + index] + partitionHistogram[RADIX * partitionIndex + index] - v;
  }
  barrier();

  // rearrange keys. grouping keys together makes dstOffset to be almost sequential, grants huge speed boost.
  // now localHistogram is unused, so alias memory.
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    localHistogram[localOffsets[i]] = localKeys[i];

#ifdef KEY_VALUE
    localHistogram[PARTITION_SIZE + localOffsets[i]] = localValues[i];
#endif
  }
  barrier();

  // binning
  for (uint i = index; i < PARTITION_SIZE; i += WORKGROUP_SIZE) {
    uint key = localHistogram[i];
    uint radix = bitfieldExtract(key, pass * 8, 8);
    uint dstOffset = localHistogramSum[radix] + i;
    if (dstOffset < elementCount) {
      outKeys[dstOffset] = key;

#ifdef KEY_VALUE
      outValues[dstOffset] = localHistogram[PARTITION_SIZE + i];
#endif
    }
  }
}

)shader";

#endif  // VK_RADIX_SORT_SHADER_DOWNSWEEP_H
