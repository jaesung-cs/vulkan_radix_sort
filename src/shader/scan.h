#ifndef VK_RADIX_SORT_SHADER_SCAN_H
#define VK_RADIX_SORT_SHADER_SCAN_H

const char* scan_comp = R"shader(
#version 460 core

const uint RADIX = 256;
const uint WORKGROUP_SIZE = 512;

// dispatch with group count (1, 1, 1)
layout (local_size_x = RADIX) in;

layout (set = 0, binding = 0) readonly buffer Histogram {
  uint histogram[];  // (4, R)
};

layout (set = 0, binding = 1) writeonly buffer HistogramCumsum {
  uint histogramCumsum[];  // (4, R)
};

// to avoid bank conflict?
const uint ROWS = RADIX + 1;
shared uint count[4 * ROWS];

void main() {
  uint index = gl_GlobalInvocationID.x;
  if (index >= RADIX) return;

  // TODO: implement parallel scan

  for (int i = 0; i < 4; ++i) count[ROWS * i + index] = histogram[RADIX * i + index];
  barrier();

  if (index < 4) {
    uint psum = 0;
    for (int i = 0; i < RADIX; ++i) {
      uint c = count[ROWS * index + i];
      count[ROWS * index + i] = psum;
      psum += c;
    }
  }
  barrier();
  
  for (int i = 0; i < 4; ++i) histogramCumsum[RADIX * i + index] = count[ROWS * i + index];
}

)shader";

#endif  // VK_RADIX_SORT_SHADER_SCAN_H
