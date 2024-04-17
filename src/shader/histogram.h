#ifndef VK_RADIX_SORT_SHADER_HISTOGRAM_H
#define VK_RADIX_SORT_SHADER_HISTOGRAM_H

const char* histogram_comp = R"shader(
#version 460 core

const uint RADIX = 256;

layout (local_size_x_id = 0) in;

layout (push_constant) uniform PushConstant {
  uint elementCount;
};

layout (set = 0, binding = 0) writeonly buffer Histogram {
  uint histogram[];  // (4, R)
};

layout (set = 1, binding = 0) readonly buffer Keys {
  uint keys[];  // (N)
};

shared uint localHistogram[4 * RADIX];  // (4, R)

void main() {
  uint localIndex = gl_LocalInvocationID.x;
  uint globalIndex = gl_GlobalInvocationID.x;

  // set local histogram zero
  for (uint i = localIndex; i < 4 * RADIX; i += gl_WorkGroupSize.x) {
    localHistogram[i] = 0;
  }
  barrier();

  // load key, add to local histogram
  if (globalIndex < elementCount) {
    uint key = keys[globalIndex];
    uint key0 = bitfieldExtract(key, 0, 8);
    uint key1 = bitfieldExtract(key, 8, 8);
    uint key2 = bitfieldExtract(key, 16, 8);
    uint key3 = bitfieldExtract(key, 24, 8);

    atomicAdd(localHistogram[RADIX * 0 + key0], 1);
    atomicAdd(localHistogram[RADIX * 1 + key1], 1);
    atomicAdd(localHistogram[RADIX * 2 + key2], 1);
    atomicAdd(localHistogram[RADIX * 3 + key3], 1);
  }
  barrier();

  // local histogram to global histogram
  for (uint i = localIndex; i < 4 * RADIX; i += gl_WorkGroupSize.x) {
    atomicAdd(histogram[i], localHistogram[i]);
  }
}

)shader";

#endif  // VK_RADIX_SORT_SHADER_HISTOGRAM_H
