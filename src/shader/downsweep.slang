import constants;

// Stride for sh0[HISTOGRAM_STRIDE * radix + waveIndex].
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

groupshared uint sh0[HISTOGRAM_STRIDE * RADIX];
groupshared uint sh1[RADIX * 2];

uint4 WaveMatch8(uint value) {
  uint4 mask = uint4(0xFFFFFFFF);
  [ForceUnroll]
  for (int j = 0; j < 8; ++j) {
    uint digit = (value >> j) & 1;
    uint4 ballot = WaveActiveBallot(digit == 1);
    // digit - 1 is 0 or 0xffffffff. xor to flip.
    mask &= uint4(digit - 1) ^ ballot;
  }
  return mask;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE)]
void main(uint3 groupThreadID: SV_GroupThreadID, uint3 groupId: SV_GroupID,
          uint groupIndex: SV_GroupIndex, uniform int pass) {
  uint elementCount = elementCounts[0];

  uint laneIndex = WaveGetLaneIndex();          // 0..31 or 0..63
  uint laneCount = WaveGetLaneCount();          // 32 or 64
  uint waveIndex = groupIndex / laneCount;      // 0..15 or 0..7
  uint waveCount = WORKGROUP_SIZE / laneCount;  // 16 or 8
  uint index = waveIndex * laneCount + laneIndex;

  uint4 waveMask = WaveGetLaneLtMask();

  uint partitionIndex = groupId.x;
  uint partitionStart = partitionIndex * PARTITION_SIZE;

  if (partitionStart >= elementCount)
    return;

  for (int i = index; i < HISTOGRAM_STRIDE * RADIX; i += WORKGROUP_SIZE) {
    sh0[i] = 0;
  }
  GroupMemoryBarrierWithGroupSync();

  // load from global memory, local histogram and offset
  uint localKeys[PARTITION_DIVISION];
  uint localRadices[PARTITION_DIVISION];
  uint localOffsets[PARTITION_DIVISION];
  uint localCounts[PARTITION_DIVISION];
#ifdef KEY_VALUE
  uint localValues[PARTITION_DIVISION];
#endif  // KEY_VALUE

  // load keys
  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    // view (4096) input partition as (:, PARTITION_DIVISION, laneCount)
    // where axis=0 for wave index, axis=1 for partition division, axis=2 for lane index.
    uint keyIndex =
        partitionStart + (PARTITION_DIVISION * laneCount) * waveIndex + i * laneCount + laneIndex;
    uint key = keyIndex < elementCount ? keysIn[keyIndex] : 0xffffffff;
    localKeys[i] = key;

#ifdef KEY_VALUE
    localValues[i] = keyIndex < elementCount ? valuesIn[keyIndex] : 0;
#endif  // KEY_VALUE

    uint radix = bitfieldExtract(key, pass * 8, 8);
    localRadices[i] = radix;
  }

  // calculate wave radix counts, offsets
  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint radix = localRadices[i];

    // mask per digit
    uint4 mask = WaveMatch8(radix);

    // wave level offset for radix
    uint waveOffset = _WaveCountBits(waveMask & mask);
    uint radixCount = _WaveCountBits(mask);

    // elect a representative per radix, add to histogram
    if (waveOffset == 0) {
      // accumulate to local histogram
      sh0[HISTOGRAM_STRIDE * radix + waveIndex] += radixCount;
      localCounts[i] = radixCount;
    } else {
      localCounts[i] = 0;
    }

    localOffsets[i] = waveOffset;
  }
  GroupMemoryBarrierWithGroupSync();

  // cumsum (256, 17), axis=1
  for (uint i = index; i < RADIX * waveCount; i += WORKGROUP_SIZE) {
    uint x = i / waveCount;
    uint y = i % waveCount;
    uint id = x * HISTOGRAM_STRIDE + y;

    uint v = sh0[id];
    uint excl = WavePrefixSum(v);
    uint prev = WaveReadLaneAt(excl, laneIndex / waveCount * waveCount);
    sh0[id] = excl - prev;
    if ((i + 1) % waveCount == 0) {
      sh1[i / waveCount] = excl + v - prev;
    }
  }
  GroupMemoryBarrierWithGroupSync();

  // cumsum (256)
  if (index < RADIX) {
    uint v = sh1[index];
    uint excl = WavePrefixSum(v);
    sh1[index] = excl;
    if (laneIndex == laneCount - 1) {
      sh1[RADIX + index / laneCount] = excl + v;
    }
  }
  GroupMemoryBarrierWithGroupSync();

  if (index < RADIX / laneCount) {
    uint v = sh1[RADIX + index];
    uint excl = WavePrefixSum(v);
    sh1[RADIX + index] = excl;
  }
  GroupMemoryBarrierWithGroupSync();

  if (index < RADIX) {
    sh1[index] += sh1[RADIX + index / laneCount];
  }
  GroupMemoryBarrierWithGroupSync();

  // calculate local offsets
  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint radix = localRadices[i];
    localOffsets[i] += sh0[HISTOGRAM_STRIDE * radix + waveIndex] + sh1[radix];

    GroupMemoryBarrierWithGroupSync();
    if (localCounts[i] > 0) {
      sh0[HISTOGRAM_STRIDE * radix + waveIndex] += localCounts[i];
    }
    GroupMemoryBarrierWithGroupSync();
  }

  if (index < RADIX) {
    sh1[index] = globalHistogram[RADIX * pass + index] + partitionHistogram[RADIX * partitionIndex + index] - sh1[index];
  }
  GroupMemoryBarrierWithGroupSync();

  // rearrange keys. grouping keys together makes dstOffset to be almost sequential, grants huge
  // speed boost. now sh0 is unused, so alias memory.
  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    sh0[localOffsets[i]] = localKeys[i];
  }
  GroupMemoryBarrierWithGroupSync();

  // binning
  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint id = i * WORKGROUP_SIZE + index;
    uint key = sh0[id];
    uint radix = bitfieldExtract(key, pass * 8, 8);
    uint dstOffset = sh1[radix] + id;
    if (dstOffset < elementCount) {
      keysOut[dstOffset] = key;
    }

#ifdef KEY_VALUE
    localKeys[id / WORKGROUP_SIZE] = dstOffset;
#endif  // KEY_VALUE
  }

#ifdef KEY_VALUE
  GroupMemoryBarrierWithGroupSync();

  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    sh0[localOffsets[i]] = localValues[i];
  }
  GroupMemoryBarrierWithGroupSync();

  [[ForceUnroll]]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint id = i * WORKGROUP_SIZE + index;
    uint value = sh0[id];
    uint dstOffset = localKeys[id / WORKGROUP_SIZE];
    if (dstOffset < elementCount) {
      valuesOut[dstOffset] = value;
    }
  }
#endif  // KEY_VALUE
}
