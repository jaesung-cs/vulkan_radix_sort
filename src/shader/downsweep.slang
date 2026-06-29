import constants;

// Stride for localHistogram[HISTOGRAM_STRIDE * radix + waveIndex].
// Must be >= max waveCount (16 for 32-lane) and coprime to 32 (bank count) to avoid bank conflicts.
static const uint HISTOGRAM_STRIDE = 17;

StructuredBuffer<uint> elementCounts : register(t0, space0);
RWStructuredBuffer<uint> globalHistogram : register(u1, space0);
RWStructuredBuffer<uint> partitionHistogram : register(u2, space0);
StructuredBuffer<uint> keysIn : register(t3, space0);
RWStructuredBuffer<uint> keysOut : register(u4, space0);
#ifdef KEY_VALUE
StructuredBuffer<uint> valuesIn : register(t5, space0);
RWStructuredBuffer<uint> valuesOut : register(u6, space0);
#endif  // KEY_VALUE

groupshared uint localHistogram[HISTOGRAM_STRIDE * RADIX];  // histogram: 17*256=4352; key scatter alias: PARTITION_SIZE=4096
groupshared uint localHistogramSum[RADIX];

// returns 0b00000....11111, where msb is laneIndex-1.
uint4 GetExclusiveWaveMask(uint laneIndex) {
    uint4 mask = uint4(0, 0, 0, 0);
    [ForceUnroll]
    for (int i = 0; i < 4; ++i) {
        if (laneIndex < 32) {
            mask[i] = (1u << laneIndex) - 1u;
            laneIndex = 0;
        } else {
            mask[i] = 0xFFFFFFFF;
            laneIndex -= 32;
        }
    }
    return mask;
}

uint GetBitCount(uint4 value) {
  uint4 result = countbits(value);
  return result[0] + result[1] + result[2] + result[3];
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE)]
void main(uint3 groupThreadID: SV_GroupThreadID, uint3 groupId: SV_GroupID,
          uint groupIndex: SV_GroupIndex, uniform int pass) {
  uint elementCount = elementCounts[0];

  uint laneIndex = WaveGetLaneIndex();          // 0..31 or 0..63
  uint laneCount = WaveGetLaneCount();          // 32 or 64
  uint waveIndex = groupIndex / laneCount;      // 0..15 or 0..7
  uint waveCount = WORKGROUP_SIZE / laneCount;  // 32 or 16
  uint index = waveIndex * laneCount + laneIndex;

  uint4 waveMask = GetExclusiveWaveMask(laneIndex);

  uint partitionIndex = groupId.x;
  uint partitionStart = partitionIndex * PARTITION_SIZE;

  if (partitionStart >= elementCount)
    return;

  if (index < RADIX) {
    for (uint i = 0; i < HISTOGRAM_STRIDE; ++i) {
      localHistogram[HISTOGRAM_STRIDE * index + i] = 0;
    }
  }
  GroupMemoryBarrierWithGroupSync();

  // load from global memory, local histogram and offset
  uint localKeys[PARTITION_DIVISION];
  uint localRadix[PARTITION_DIVISION];
  uint localOffsets[PARTITION_DIVISION];
  uint waveHistogram[PARTITION_DIVISION];
#ifdef KEY_VALUE
  uint localValues[PARTITION_DIVISION];
#endif  // KEY_VALUE

  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint keyIndex =
        partitionStart + (PARTITION_DIVISION * laneCount) * waveIndex + i * laneCount + laneIndex;
    uint key = keyIndex < elementCount ? keysIn[keyIndex] : 0xffffffff;
    localKeys[i] = key;

#ifdef KEY_VALUE
    localValues[i] = keyIndex < elementCount ? valuesIn[keyIndex] : 0;
#endif  // KEY_VALUE

    uint radix = bitfieldExtract(key, pass * 8, 8);
    localRadix[i] = radix;

    // mask per digit
    uint4 mask = WaveActiveBallot(true);
    [ForceUnroll]
    for (int j = 0; j < 8; ++j) {
      uint digit = (radix >> j) & 1;
      uint4 ballot = WaveActiveBallot(digit == 1);
      // digit - 1 is 0 or 0xffffffff. xor to flip.
      mask &= uint4(digit - 1) ^ ballot;
    }

    // wave level offset for radix
    uint waveOffset = GetBitCount(waveMask & mask);
    uint radixCount = GetBitCount(mask);

    // elect a representative per radix, add to histogram
    if (waveOffset == 0) {
      // accumulate to local histogram
      __atomic_add(localHistogram[HISTOGRAM_STRIDE * radix + waveIndex], radixCount, MemoryOrder.Relaxed);
      waveHistogram[i] = radixCount;
    } else {
      waveHistogram[i] = 0;
    }

    localOffsets[i] = waveOffset;
  }
  GroupMemoryBarrierWithGroupSync();

  // local histogram reduce; padding slots are 0 and don't affect prefix values
  for (uint i = index; i < HISTOGRAM_STRIDE * RADIX; i += WORKGROUP_SIZE) {
    uint v = localHistogram[i];
    uint sum = WaveActiveSum(v);
    uint excl = WavePrefixSum(v);
    localHistogram[i] = excl;
    if (laneIndex == 0) {
      localHistogramSum[i / laneCount] = sum;
    }
  }
  GroupMemoryBarrierWithGroupSync();

  // local histogram reduce
  uint intermediateOffset0 = HISTOGRAM_STRIDE * RADIX / laneCount;
  if (index < intermediateOffset0) {
    uint v = localHistogramSum[index];
    uint sum = WaveActiveSum(v);
    uint excl = WavePrefixSum(v);
    localHistogramSum[index] = excl;
    if (laneIndex == 0) {
      localHistogramSum[intermediateOffset0 + index / laneCount] = sum;
    }
  }
  GroupMemoryBarrierWithGroupSync();

  // local histogram reduce (ceiling division to cover partial last wave)
  uint intermediateSize1 = max((intermediateOffset0 + laneCount - 1) / laneCount, 1u);
  if (index < intermediateSize1) {
    uint v = localHistogramSum[intermediateOffset0 + index];
    uint excl = WavePrefixSum(v);
    localHistogramSum[intermediateOffset0 + index] = excl;
  }
  GroupMemoryBarrierWithGroupSync();

  // local histogram add 128
  if (index < intermediateOffset0) {
    localHistogramSum[index] += localHistogramSum[intermediateOffset0 + index / laneCount];
  }
  GroupMemoryBarrierWithGroupSync();

  // local histogram add
  for (uint i = index; i < HISTOGRAM_STRIDE * RADIX; i += WORKGROUP_SIZE) {
    localHistogram[i] += localHistogramSum[i / laneCount];
  }
  GroupMemoryBarrierWithGroupSync();

  // post-scan stage
  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint radix = localRadix[i];
    localOffsets[i] += localHistogram[HISTOGRAM_STRIDE * radix + waveIndex];

    GroupMemoryBarrierWithGroupSync();
    if (waveHistogram[i] > 0) {
      __atomic_add(localHistogram[HISTOGRAM_STRIDE * radix + waveIndex], waveHistogram[i],
                   MemoryOrder.Relaxed);
    }
    GroupMemoryBarrierWithGroupSync();
  }

  // after atomicAdd, localHistogram contains inclusive sum
  if (index < RADIX) {
    uint v = index == 0 ? 0 : localHistogram[HISTOGRAM_STRIDE * (index - 1) + waveCount - 1];
    localHistogramSum[index] = globalHistogram[RADIX * pass + index] +
                               partitionHistogram[RADIX * partitionIndex + index] - v;
  }
  GroupMemoryBarrierWithGroupSync();

  // rearrange keys. grouping keys together makes dstOffset to be almost sequential, grants huge
  // speed boost. now localHistogram is unused, so alias memory.
  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    localHistogram[localOffsets[i]] = localKeys[i];
  }
  GroupMemoryBarrierWithGroupSync();

  // binning
  for (uint i = index; i < PARTITION_SIZE; i += WORKGROUP_SIZE) {
    uint key = localHistogram[i];
    uint radix = bitfieldExtract(key, pass * 8, 8);
    uint dstOffset = localHistogramSum[radix] + i;
    if (dstOffset < elementCount) {
      keysOut[dstOffset] = key;
    }

#ifdef KEY_VALUE
    localKeys[i / WORKGROUP_SIZE] = dstOffset;
#endif  // KEY_VALUE
  }

#ifdef KEY_VALUE
  GroupMemoryBarrierWithGroupSync();

  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    localHistogram[localOffsets[i]] = localValues[i];
  }
  GroupMemoryBarrierWithGroupSync();

  for (uint i = index; i < PARTITION_SIZE; i += WORKGROUP_SIZE) {
    uint value = localHistogram[i];
    uint dstOffset = localKeys[i / WORKGROUP_SIZE];
    if (dstOffset < elementCount) {
      valuesOut[dstOffset] = value;
    }
  }
#endif  // KEY_VALUE
}
