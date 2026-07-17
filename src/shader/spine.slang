import constants;

ByteAddressBuffer elementCounts : register(t0, space0);
RWByteAddressBuffer globalHistogram : register(u1, space0);
RWByteAddressBuffer partitionHistogram : register(u2, space0);

groupshared uint reduction;
groupshared uint sh0[MAX_SUBGROUP_SIZE];
groupshared uint sh1[RADIX];

// dispatch this shader (RADIX, 1, 1), so that gl_WorkGroupID.x is radix
[shader("compute")]
[numthreads(WORKGROUP_SIZE)]
void main(uint3 groupId: SV_GroupID, uint groupIndex: SV_GroupIndex, uniform int pass) {
  uint elementCount = elementCounts.Load<uint>(0);

  uint laneIndex = WaveGetLaneIndex();  // 0..31 or 0..63
  uint laneCount = WaveGetLaneCount();  // 32 or 64
  uint waveIndex = groupIndex / laneCount;
  uint waveCount = WORKGROUP_SIZE / laneCount;
  uint index = waveIndex * laneCount + laneIndex;

  uint radix = groupId.x;

  uint partitionCount = (elementCount + PARTITION_SIZE - 1) / PARTITION_SIZE;

  if (index == 0) {
    reduction = 0;
  }
  GroupMemoryBarrierWithGroupSync();

  for (uint i = index; i < partitionCount; i += WORKGROUP_SIZE) {
    uint value = i < partitionCount
                     ? partitionHistogram.Load<uint>((partitionCount * radix + i) * sizeof(uint))
                     : 0;
    uint excl = WavePrefixSum(value) + reduction;

    if (laneIndex == laneCount - 1) {
      sh0[waveIndex] = excl + value - reduction;
    }
    GroupMemoryBarrierWithGroupSync();

    if (index < waveCount) {
      uint v = sh0[index];
      uint excl = WavePrefixSum(v);
      sh0[index] = excl;

      if (index == waveCount - 1) {
        reduction += excl + v;
      }
    }
    GroupMemoryBarrierWithGroupSync();

    if (i < partitionCount) {
      partitionHistogram.Store<uint>((partitionCount * radix + i) * sizeof(uint),
                                     excl + sh0[waveIndex]);
    }
  }

  if (radix == 0) {
    // one workgroup is responsible for global histogram prefix sum
    if (index < RADIX) {
      uint value = globalHistogram.Load<uint>((RADIX * pass + index) * sizeof(uint));
      uint excl = WavePrefixSum(value);

      if (laneIndex == laneCount - 1) {
        sh1[waveIndex] = excl + value;
      }
      GroupMemoryBarrierWithGroupSync();

      if (index < RADIX / laneCount) {
        sh1[index] = WavePrefixSum(sh1[index]);
      }
      GroupMemoryBarrierWithGroupSync();

      globalHistogram.Store<uint>((RADIX * pass + index) * sizeof(uint), excl + sh1[waveIndex]);
    }
  }
}
