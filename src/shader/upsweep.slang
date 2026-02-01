import constants;

struct Data {
  uint pass;
  uint* elementCounts;
  uint* globalHistogram;
  uint* partitionHistogram;
  uint* keysIn;
};

groupshared uint localHistogram[RADIX];

[shader("compute")]
[numthreads(WORKGROUP_SIZE)]
void main(uint3 groupThreadID: SV_GroupThreadID, uint3 groupId: SV_GroupID, uniform Data data) {
  uint elementCount = data.elementCounts[0];

  uint index = groupThreadID.x;
  uint partitionIndex = groupId.x;
  uint partitionStart = partitionIndex * PARTITION_SIZE;

  // discard all workgroup invocations
  if (partitionStart >= elementCount) {
    return;
  }

  if (index < RADIX) {
    localHistogram[index] = 0;
  }
  GroupMemoryBarrierWithGroupSync();

  // local histogram
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint keyIndex = partitionStart + WORKGROUP_SIZE * i + index;
    uint key = keyIndex < elementCount ? data.keysIn[keyIndex] : 0xffffffff;
    uint radix = bitfieldExtract(key, 8 * data.pass, 8);
    __atomic_add(localHistogram[radix], 1, MemoryOrder.Relaxed);
  }
  GroupMemoryBarrierWithGroupSync();

  if (index < RADIX) {
    // set to partition histogram
    data.partitionHistogram[RADIX * partitionIndex + index] = localHistogram[index];

    // add to global histogram
    __atomic_add(data.globalHistogram[RADIX * data.pass + index], localHistogram[index], MemoryOrder.Relaxed);
  }
}
