#ifndef VK_RADIX_SORT_SHADER_HISTOGRAM_H
#define VK_RADIX_SORT_SHADER_HISTOGRAM_H

const char* histogram_comp = R"shader(
#version 460 core

const uint RADIX = 256;
const uint WORKGROUP_SIZE = 512;

layout (local_size_x = WORKGROUP_SIZE) in;

layout (push_constant) uniform PushConstant {
  uint elementCount;
};

layout (set = 0, binding = 0) writeonly buffer Histogram {
  uint histogram[];  // (4, R)
};

layout (set = 1, binding = 0) readonly buffer Keys {
  uint keys[];  // (N)
};

void main() {
  uint index = gl_GlobalInvocationID.x;
  if (index >= elementCount) return;

  uint key = keys[index];
  uint key0 = bitfieldExtract(key, 0, 8);
  uint key1 = bitfieldExtract(key, 8, 8);
  uint key2 = bitfieldExtract(key, 16, 8);
  uint key3 = bitfieldExtract(key, 24, 8);

  // TODO: use shared memory for atomic counter
  atomicAdd(histogram[RADIX * 0 + key0], 1);
  atomicAdd(histogram[RADIX * 1 + key1], 1);
  atomicAdd(histogram[RADIX * 2 + key2], 1);
  atomicAdd(histogram[RADIX * 3 + key3], 1);
}

)shader";

#endif  // VK_RADIX_SORT_SHADER_HISTOGRAM_H
