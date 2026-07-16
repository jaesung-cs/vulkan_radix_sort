import constants;

ByteAddressBuffer elementCounts : register(t0, space0);
RWByteAddressBuffer globalHistogram : register(u1, space0);
RWByteAddressBuffer partitionHistogram : register(u2, space0);
ByteAddressBuffer keys : register(t3, space0);

groupshared uint localHistogram[RADIX];

[shader("compute")]
[numthreads(WORKGROUP_SIZE)]
void main(uint3 groupThreadID: SV_GroupThreadID, uint3 groupId: SV_GroupID, uniform int pass) {
  uint elementCount = elementCounts.Load<uint>(0);

  uint index = groupThreadID.x;
  uint partitionIndex = groupId.x;
  uint partitionStart = partitionIndex * PARTITION_SIZE;

  if (index < RADIX) {
    localHistogram[index] = 0;
  }
  GroupMemoryBarrierWithGroupSync();

  // local histogram
  [ForceUnroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint keyIndex = partitionStart + WORKGROUP_SIZE * i + index;
    uint key = keyIndex < elementCount ? keys.Load<uint>(keyIndex * sizeof(uint)) : 0xffffffff;
    uint radix = bitfieldExtract(key, 8 * pass, 8);
    InterlockedAdd(localHistogram[radix], 1);
  }
  GroupMemoryBarrierWithGroupSync();

  if (index < RADIX) {
    // set to partition histogram
    partitionHistogram.Store<uint>((RADIX * partitionIndex + index) * sizeof(uint), localHistogram[index]);

    // add to global histogram
    globalHistogram.InterlockedAdd((RADIX * pass + index) * sizeof(uint), localHistogram[index]);
  }
}
