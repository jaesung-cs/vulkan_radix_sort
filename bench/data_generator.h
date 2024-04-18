#ifndef VK_RADIX_SORT_DATA_GENERATOR_H
#define VK_RADIX_SORT_DATA_GENERATOR_H

#include <vector>

struct SortData {
  std::vector<uint32_t> keys;
  std::vector<uint32_t> values;
};

SortData GenerateUniformRandomData(uint32_t size, uint32_t bits = 32);

#endif  // VK_RADIX_SORT_DATA_GENERATOR_H
