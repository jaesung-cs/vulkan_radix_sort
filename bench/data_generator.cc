#include "data_generator.h"

#include <random>

SortData GenerateUniformRandomData(uint32_t size, uint32_t bits) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dist_values;
  std::uniform_int_distribution<uint32_t> dist_keys;
  if (bits < 32)
    dist_keys = std::uniform_int_distribution<uint32_t>(0, (1u << bits) - 1);

  SortData data;
  data.keys.reserve(size);
  data.values.reserve(size);
  for (int i = 0; i < size; i++) {
    data.keys.push_back(dist_keys(gen));
  }
  for (int i = 0; i < size; i++) {
    data.values.push_back(dist_values(gen));
  }
  return data;
}
