#ifndef VK_RADIX_SORT_SHADER_BINNING_H
#define VK_RADIX_SORT_SHADER_BINNING_H

const char* binning_comp = R"shader(
#version 460 core

#extension GL_KHR_shader_subgroup_basic: enable
#extension GL_KHR_shader_subgroup_ballot: enable
#extension GL_KHR_shader_subgroup_shuffle: enable
#extension GL_KHR_shader_subgroup_arithmetic: enable

const uint RADIX = 256;
const uint WORKGROUP_SIZE = 512;
// subgroup size is 32, 64, or 128.
const uint MAX_SUBGROUP_COUNT = 512 / 32;
const uint WORKGROUP_COUNT = 15;
const uint PARTITION_SIZE = WORKGROUP_SIZE * WORKGROUP_COUNT;

layout (local_size_x = WORKGROUP_SIZE) in;

layout (push_constant) uniform PushConstant {
  uint elementCount;
  int pass;
};

layout (set = 0, binding = 0) readonly buffer Histogram {
  uint histogram[];  // (4, R)
};

layout (set = 0, binding = 1, std430) buffer Lookback {
  uint partitionCounter;  // startin from 0

  // Volatile memory enables lookback!
  volatile uint lookback[];  // (ceil(N/P), R)
};

layout (set = 1, binding = 0) readonly buffer Keys {
  uint keys[];  // (N)
};

layout (set = 1, binding = 1) writeonly buffer OutKeys {
  uint outKeys[];  // (N)
};

// Onesweep lookback status. 0x3 = 0b11 for GLOBAL_SUM, for |(or) operator.
#define LOCAL_COUNT 0x1
#define GLOBAL_SUM 0x3

shared uint partitionIndex;

shared uint localHistogram[PARTITION_SIZE];  // (R, S=16), or (P) for alias.
shared int localHistogramSum[RADIX];  // (R)
// TODO: further alias scan intermediate
shared uint scanIntermediate[RADIX * MAX_SUBGROUP_COUNT / (32 - 1)];  // 1/n + 1/n^2 + ... < 1/(n-1)

// returns 0b00000....11111, where msb is id-1.
uvec4 GetExclusiveSubgroupMask(uint id) {
  return uvec4(
    (1 << id) - 1,
    id < 32 ? 0 : (1 << (id - 32)) - 1,
    id < 64 ? 0 : (1 << (id - 64)) - 1,
    id < 96 ? 0 : (1 << (id - 96)) - 1
  );
}

uint GetBitCount(uvec4 value) {
  uvec4 result = bitCount(value);
  return result[0] + result[1] + result[2] + result[3];
}

void main() {
  uint threadIndex = gl_SubgroupInvocationID;  // 0..31
  uvec4 subgroupMask = GetExclusiveSubgroupMask(threadIndex);
  uint subgroupIndex = gl_SubgroupID;  // 0..15
  uint index = subgroupIndex * gl_SubgroupSize + threadIndex;

  // initialize shared variables
  // the workgroup is responsible for partitionIndex
  if (index == 0) {
    partitionIndex = atomicAdd(partitionCounter, 1);
  }
  if (index < RADIX) {
    localHistogramSum[index] = 0;
    for (int i = 0; i < gl_NumSubgroups; ++i) {
      localHistogram[gl_NumSubgroups * index + i] = 0;
    }
  }
  barrier();

  // load from global memory, local histogram and offset
  uint localKeys[WORKGROUP_COUNT];
  uint localOffsets[WORKGROUP_COUNT];
  uint subgroupHistogram[WORKGROUP_COUNT];
  for (int i = 0; i < WORKGROUP_COUNT; ++i) {
    uint keyIndex = PARTITION_SIZE * partitionIndex + (WORKGROUP_COUNT * gl_SubgroupSize) * subgroupIndex + i * gl_SubgroupSize + threadIndex;
    uint key = keyIndex < elementCount ? keys[keyIndex] : 0xffffffff;
    localKeys[i] = key;

    uint radix = bitfieldExtract(key, pass * 8, 8);

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

    // elect a representative per radix, adds to histogram
    if (subgroupOffset == 0) {
      // accumulate to local histogram
      atomicAdd(localHistogram[gl_NumSubgroups * radix + subgroupIndex], radixCount);
      subgroupHistogram[i] = radixCount;
    } else {
      subgroupHistogram[i] = 0;
    }

    localOffsets[i] = subgroupOffset;
  }
  barrier();

  // local histogram reduce 4096
  for (uint i = index; i < RADIX * gl_NumSubgroups; i += WORKGROUP_SIZE) {
    uint v = localHistogram[i];
    uint sum = subgroupAdd(v);
    uint excl = subgroupExclusiveAdd(v);
    localHistogram[i] = excl;
    if (threadIndex == 0) {
      scanIntermediate[i / gl_SubgroupSize] = sum;
    }
  }
  barrier();

  // local histogram reduce 128
  uint intermediateOffset0 = RADIX * gl_NumSubgroups / gl_SubgroupSize;
  if (index < intermediateOffset0) {
    uint v = scanIntermediate[index];
    uint sum = subgroupAdd(v);
    uint excl = subgroupExclusiveAdd(v);
    scanIntermediate[index] = excl;
    if (threadIndex == 0) {
      scanIntermediate[intermediateOffset0 + index / gl_SubgroupSize] = sum;
    }
  }
  barrier();

  // local histogram reduce 4
  uint intermediateOffset1 = RADIX * gl_NumSubgroups / gl_SubgroupSize / gl_SubgroupSize;
  if (index < intermediateOffset1) {
    uint v = scanIntermediate[intermediateOffset0 + index];
    uint excl = subgroupExclusiveAdd(v);
    scanIntermediate[intermediateOffset0 + index] = excl;
  }
  barrier();

  // local histogram add 128
  if (index < intermediateOffset0) {
    scanIntermediate[index] += scanIntermediate[intermediateOffset0 + index / gl_SubgroupSize];
  }
  barrier();

  // local histogram add 4096
  for (uint i = index; i < RADIX * gl_NumSubgroups; i += WORKGROUP_SIZE) {
    localHistogram[i] += scanIntermediate[i / gl_SubgroupSize];
  }
  barrier();

  // post-scan stage
  for (int i = 0; i < WORKGROUP_COUNT; ++i) {
    uint key = localKeys[i];
    uint radix = bitfieldExtract(key, pass * 8, 8);
    localOffsets[i] += localHistogram[gl_NumSubgroups * radix + subgroupIndex];
    // TODO: remove barrier
    barrier();
    atomicAdd(localHistogram[gl_NumSubgroups * radix + subgroupIndex], subgroupHistogram[i]);
    barrier();
  }

  // after atomicAdd, localHistogram contains inclusive sum
  // update lookback
  uint localCount = 0;
  if (index < RADIX) {
    localCount = localHistogram[gl_NumSubgroups * (index + 1) - 1];
    if (index > 0) {
      localCount -= localHistogram[gl_NumSubgroups * index - 1];
    }
    lookback[RADIX * partitionIndex + index] = localCount | (LOCAL_COUNT << 30);
  }
  barrier();

  // inclusive sum with lookback
  if (index < RADIX) {
    if (partitionIndex == 0) {
      lookback[RADIX * partitionIndex + index] |= (GLOBAL_SUM << 30);
      localHistogramSum[index] = int(localCount) - int(localHistogram[gl_NumSubgroups * (index + 1) - 1]);
    } else {
      // lookback
      uint globalSum = localCount;
      int lookbackIndex = int(partitionIndex) - 1;
      while (lookbackIndex >= 0) {
        uint lookbackValue = lookback[RADIX * lookbackIndex + index];
        uint status = bitfieldExtract(lookbackValue, 30, 2);

        if (status == GLOBAL_SUM) {
          globalSum += bitfieldExtract(lookbackValue, 0, 30);
          break;
        }

        else if (status == LOCAL_COUNT) {
          globalSum += bitfieldExtract(lookbackValue, 0, 30);
          lookbackIndex--;
        }

        // if not ready, lookback again
      }

      // update global sum
      lookback[RADIX * partitionIndex + index] = globalSum | (GLOBAL_SUM << 30);

      // store exclusive sum
      localHistogramSum[index] = int(globalSum) - int(localHistogram[gl_NumSubgroups * (index + 1) - 1]);
    }
  }
  barrier();

  // rearrange keys. now localHistogram is unused, so alias memory.
  for (int i = 0; i < WORKGROUP_COUNT; ++i) {
    localHistogram[localOffsets[i]] = localKeys[i];
  }
  barrier();

  // binning
  for (uint i = index; i < PARTITION_SIZE; i += WORKGROUP_SIZE) {
    uint key = localHistogram[i];
    uint radix = bitfieldExtract(key, pass * 8, 8);
    uint dstOffset = histogram[RADIX * pass + radix] + localHistogramSum[radix] + i;
    if (dstOffset < elementCount) {
      outKeys[dstOffset] = key;
    }
  }
}

)shader";

#endif  // VK_RADIX_SORT_SHADER_BINNING_H
