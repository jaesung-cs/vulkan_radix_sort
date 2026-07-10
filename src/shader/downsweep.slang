import constants;

StructuredBuffer<uint> elementCounts : register(t0, space0);
RWStructuredBuffer<uint> globalHistogram : register(u1, space0);
RWStructuredBuffer<uint> partitionHistogram : register(u2, space0);
StructuredBuffer<uint> keysIn : register(t3, space0);
RWStructuredBuffer<uint> keysOut : register(u4, space0);
#ifdef KEY_VALUE
StructuredBuffer<uint> valuesIn : register(t5, space0);
RWStructuredBuffer<uint> valuesOut : register(u6, space0);
#endif  // KEY_VALUE

groupshared uint sh0[RADIX4 * 128];
groupshared uint sh1[PARTITION_SIZE];
groupshared uint sh2[RADIX];

uint4 WaveMatch4(uint value) {
  uint4 mask = uint4(0xffffffff);
  [ForceUnroll]
  for (int j = 0; j < 4; ++j) {
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

  // keys = (H, laneCount)
  // workgroup loads (i * waveCount + waveIndex, laneIndex) for i=0..P-1
  // sh0 = (R4, H)  R4 = 16
  uint H = PARTITION_SIZE / laneCount;  // 128 or 64

  uint partitionIndex = groupId.x;
  uint partitionStart = partitionIndex * PARTITION_SIZE;

  if (index < RADIX) {
    sh2[index] = globalHistogram[RADIX * pass + index] + partitionHistogram[RADIX * partitionIndex + index];
  }
  for (int i = index; i < RADIX4 * H; i += WORKGROUP_SIZE) sh0[i] = 0;
  GroupMemoryBarrierWithGroupSync();

  uint localKeys[PARTITION_DIVISION];
  uint localOffsets[PARTITION_DIVISION];
#ifdef KEY_VALUE
  uint localValues[PARTITION_DIVISION];
#endif  // KEY_VALUE

  // load keys from global memory
  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint keyIndex = partitionStart + i * WORKGROUP_SIZE + index;
    uint key = keyIndex < elementCount ? keysIn[keyIndex] : 0xffffffff;
    localKeys[i] = key;

#ifdef KEY_VALUE
    localValues[i] = keyIndex < elementCount ? valuesIn[keyIndex] : 0;
#endif  // KEY_VALUE
  }

  // calculate wave radix counts, offsets
  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint key = localKeys[i];
    uint radix = bitfieldExtract(key, pass * 8 + 0, 4);

    // mask per digit
    uint4 mask = WaveMatch4(radix);

    // wave level offset for radix
    uint waveOffset = _WaveCountBits(waveMask & mask);
    uint radixCount = _WaveCountBits(mask);

    localOffsets[i] = waveOffset;
    if (waveOffset == 0) sh0[radix * H + i * waveCount + waveIndex] += radixCount;
  }
  GroupMemoryBarrierWithGroupSync();

  // cumsum (16, 128) = (2048)
  for (uint i = index; i < RADIX4 * H; i += WORKGROUP_SIZE) {
    uint v = sh0[i];
    uint excl = WavePrefixSum(v);
    sh0[i] = excl;
    if (i % laneCount == laneCount - 1)
      sh1[i / laneCount] = excl + v;
  }
  GroupMemoryBarrierWithGroupSync();

  // cumsum (64)
  if (waveIndex == 0) {
    uint v0 = sh1[index];
    uint v1 = sh1[index + laneCount];
    uint excl0 = WavePrefixSum(v0);
    uint excl1 = WavePrefixSum(v1);
    uint sum0 = WaveReadLaneAt(excl0 + v0, laneCount - 1);
    sh1[index] = excl0;
    sh1[index + laneCount] = sum0 + excl1;
  }
  GroupMemoryBarrierWithGroupSync();

  for (uint i = index; i < RADIX4 * H; i += WORKGROUP_SIZE) sh0[i] += sh1[i / laneCount];
  GroupMemoryBarrierWithGroupSync();

  // scatter keys to sh1
  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint key = localKeys[i];
    uint radix = bitfieldExtract(key, pass * 8 + 0, 4);
    localOffsets[i] += sh0[radix * H + i * waveCount + waveIndex];
    sh1[localOffsets[i]] = key;
  }
  GroupMemoryBarrierWithGroupSync();

  // reset sh0
  for (int i = index; i < RADIX4 * H; i += WORKGROUP_SIZE) sh0[i] = 0;
  
  // reload keys
  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) localKeys[i] = sh1[i * WORKGROUP_SIZE + index];

  // scatter values to sh1
#ifdef KEY_VALUE
  // only needed here to guard the sh1 write below; skip it in keys-only builds
  // since nothing touches sh1 again until the next pass's own barrier.
  GroupMemoryBarrierWithGroupSync();

  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) sh1[localOffsets[i]] = localValues[i];
  GroupMemoryBarrierWithGroupSync();

  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) localValues[i] = sh1[i * WORKGROUP_SIZE + index];
#endif  // KEY_VALUE

  GroupMemoryBarrierWithGroupSync();

  // calculate wave radix counts, offsets
  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint key = localKeys[i];
    uint radix = bitfieldExtract(key, pass * 8 + 4, 4);

    // mask per digit
    uint4 mask = WaveMatch4(radix);

    // wave level offset for radix
    uint waveOffset = _WaveCountBits(waveMask & mask);
    uint radixCount = _WaveCountBits(mask);

    localOffsets[i] = waveOffset;
    if (waveOffset == 0) sh0[radix * H + i * waveCount + waveIndex] += radixCount;
  }
  GroupMemoryBarrierWithGroupSync();

  // cumsum (16, 128) = (2048)
  for (uint i = index; i < RADIX4 * H; i += WORKGROUP_SIZE) {
    uint v = sh0[i];
    uint excl = WavePrefixSum(v);
    sh0[i] = excl;
    if (i % laneCount == laneCount - 1) sh1[i / laneCount] = excl + v;
  }
  GroupMemoryBarrierWithGroupSync();

  // cumsum (64)
  if (waveIndex == 0) {
    uint v0 = sh1[index];
    uint v1 = sh1[index + laneCount];
    uint excl0 = WavePrefixSum(v0);
    uint excl1 = WavePrefixSum(v1);
    uint sum0 = WaveReadLaneAt(excl0 + v0, laneCount - 1);
    sh1[index] = excl0;
    sh1[index + laneCount] = sum0 + excl1;
  }
  GroupMemoryBarrierWithGroupSync();

  for (uint i = index; i < RADIX4 * H; i += WORKGROUP_SIZE) sh0[i] += sh1[i / laneCount];
  GroupMemoryBarrierWithGroupSync();

  // scatter keys to sh1
  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint key = localKeys[i];
    uint radix = bitfieldExtract(key, pass * 8 + 4, 4);
    localOffsets[i] += sh0[radix * H + i * waveCount + waveIndex];
    sh1[localOffsets[i]] = key;
  }
  GroupMemoryBarrierWithGroupSync();

  // reload keys, and find each full-byte radix's local start position in the
  // now fully (stably) sorted sh1 by detecting boundaries between neighbors.
  // this replaces the separate atomic histogram pass that used to run before
  // the sort; reuse sh0 (no longer needed after the scatter above) to hold it.
  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint id = i * WORKGROUP_SIZE + index;
    uint key = sh1[id];
    uint radix = bitfieldExtract(key, pass * 8, 8);
    uint prevRadix = bitfieldExtract(id == 0 ? key : sh1[id - 1], pass * 8, 8);
    if (id == 0 || radix != prevRadix) sh0[radix] = id;
    localKeys[i] = key;
  }
  GroupMemoryBarrierWithGroupSync();

  // scatter values to sh1
#ifdef KEY_VALUE
  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) sh1[localOffsets[i]] = localValues[i];
  GroupMemoryBarrierWithGroupSync();

  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) localValues[i] = sh1[i * WORKGROUP_SIZE + index];
  GroupMemoryBarrierWithGroupSync();
#endif  // KEY_VALUE

  // binning
  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint id = i * WORKGROUP_SIZE + index;
    uint key = localKeys[i];
    uint radix = bitfieldExtract(key, pass * 8, 8);
    uint dstOffset = sh2[radix] - sh0[radix] + id;
    if (dstOffset < elementCount) {
      keysOut[dstOffset] = key;
#ifdef KEY_VALUE
      valuesOut[dstOffset] = localValues[i];
#endif  // KEY_VALUE
    }
  }
}
