#ifndef VK_RADIX_SORT_DATA_GENERATOR_H
#define VK_RADIX_SORT_DATA_GENERATOR_H

#include <vector>

std::vector<uint32_t> GenerateUniformRandomData(uint32_t size,
                                                uint32_t bits = 32);

#endif  // VK_RADIX_SORT_DATA_GENERATOR_H
