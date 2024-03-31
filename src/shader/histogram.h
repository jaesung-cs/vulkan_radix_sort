#ifndef VK_RADIX_SORT_SHADER_HISTOGRAM_H
#define VK_RADIX_SORT_SHADER_HISTOGRAM_H

const char* histogram_comp = R"shader(
#version 460 core

layout (local_size_x = 256) in;

layout (push_constant, std430) uniform PushConstants {
  uint num_elements;
};

layout (set = 0, binding = 0) readonly buffer Keys {
  uint keys[];
};

layout (set = 0, binding = 2) writeonly buffer Histogram {
  uint histogram[];
};

void main() {
  uint index = gl_GlobalInvocationID.x;
  if (index >= num_elements) return;

  uint key = keys[index];
  atomicAdd(histogram[key & 0xFF], 1);
}

)shader";

#endif  // VK_RADIX_SORT_SHADER_HISTOGRAM_H
