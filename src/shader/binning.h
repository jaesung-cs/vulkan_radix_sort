#ifndef VK_RADIX_SORT_SHADER_BINNING_H
#define VK_RADIX_SORT_SHADER_BINNING_H

const char* binning_comp = R"shader(
#version 460 core

#extension GL_KHR_shader_subgroup_basic: enable
#extension GL_KHR_shader_subgroup_ballot: enable
#extension GL_KHR_shader_subgroup_shuffle: enable

const uint RADIX = 256;
const uint WORKGROUP_SIZE = 512;
const uint WORKGROUP_COUNT = 15;
const uint PARTITION_SIZE = WORKGROUP_SIZE * WORKGROUP_COUNT;

layout (local_size_x = WORKGROUP_SIZE) in;

layout (push_constant) uniform PushConstant {
  uint elementCount;
  int pass;
};

layout (set = 0, binding = 1) readonly buffer HistogramCumsum {
  uint histogramCumsum[];  // (4, R)
};

layout (set = 0, binding = 2, std430) buffer Lookback {
  uint partitionCounter;  // startin from 0
  uint lookback[];  // (ceil(N/P), R)
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
shared uint partitionSize;
shared uint localHistogram[RADIX];  // (R)

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

uint GetLSB(uvec4 value) {
  if (value[0] != 0) return findLSB(value[0]);
  if (value[1] != 0) return findLSB(value[1]);
  if (value[2] != 0) return findLSB(value[2]);
  if (value[3] != 0) return findLSB(value[3]);
  return -1;
}

void main() {
  uint subgroupIndex = gl_SubgroupInvocationID;
  uvec4 subgroupMask = GetExclusiveSubgroupMask(subgroupIndex);
  uint threadIndex = gl_SubgroupID * gl_SubgroupSize + subgroupIndex;

  // initialize shared variables
  // the workgroup is responsible for partitionIndex
  if (threadIndex == 0) {
    partitionIndex = atomicAdd(partitionCounter, 1);
    partitionSize = min(PARTITION_SIZE * (partitionIndex + 1) - 1, elementCount);
  }
  if (threadIndex < RADIX) {
    localHistogram[threadIndex] = 0;
  }
  barrier();

  // load from global memory, local histogram and offset
  uint localKeys[WORKGROUP_COUNT];
  uint localOffsets[WORKGROUP_COUNT];
  #pragma unroll
  for (int i = 0; i < WORKGROUP_COUNT; ++i) {
    uint keyIndex = PARTITION_SIZE * partitionIndex + WORKGROUP_SIZE * i + threadIndex;
    uint key = keyIndex < elementCount ? keys[keyIndex] : 0xffffffff;
    localKeys[i] = key;

    uint radix = bitfieldExtract(key, pass * 8, 8);

    // mask per digit
    uvec4 mask = uvec4(0xffffffff);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint digit = (radix >> j) & 1;
      uvec4 ballot = subgroupBallot(digit == 1);
      // digit - 1 is 0 or 0xffffffff. xor to flip.
      mask &= uvec4(digit - 1) ^ ballot;
    }

    // subgroup level offset for radix
    uint subgroupOffset = GetBitCount(subgroupMask & mask);

    // elect a representative per radix, adds to histogram, broadcast to invocation with same radix
    uint shuffleValue = 0;
    if (subgroupOffset == 0) {
      // accumulate to local histogram
      uint radixCount = GetBitCount(mask);
      shuffleValue = atomicAdd(localHistogram[radix], radixCount);
    }
    uint shuffleIndex = GetLSB(mask);
    uint base = subgroupShuffle(shuffleValue, shuffleIndex);

    localOffsets[i] = base + subgroupOffset;
  }
  barrier();

  // inclusive sum with lookback
  if (threadIndex < RADIX) {
    uint localValue = localHistogram[threadIndex];
    uint globalSum = localValue;

    // update local count to lookback table
    lookback[RADIX * partitionIndex + threadIndex] = globalSum | (LOCAL_COUNT << 30);

    if (partitionIndex == 0) {
      lookback[threadIndex] |= (GLOBAL_SUM << 30);
    } else {
      // lookback
      int lookbackIndex = int(partitionIndex) - 1;
      while (lookbackIndex >= 0) {
        uint lookbackValue = lookback[RADIX * lookbackIndex + threadIndex];
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
      lookback[RADIX * partitionIndex + threadIndex] = globalSum | (GLOBAL_SUM << 30);
    }
    
    // now globalSum stores global inclusive sum. store exclusive sum back to shared variable.
    localHistogram[threadIndex] = globalSum - localValue;
  }
  barrier();

  // binning
  for (int i = 0; i < WORKGROUP_COUNT; ++i) {
    uint key = localKeys[i];
    uint radix = bitfieldExtract(key, pass * 8, 8);
    uint dstOffset = histogramCumsum[RADIX * pass + radix] + localOffsets[i];
    if (dstOffset < elementCount) {
      outKeys[dstOffset] = key;
    }
  }
}

)shader";

#endif  // VK_RADIX_SORT_SHADER_BINNING_H
