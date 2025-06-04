#include "data_generator.h"

DataGenerator::DataGenerator() {
  std::random_device rd;
  gen_ = std::mt19937(rd());
}

DataGenerator::DataGenerator(int seed) : gen_(seed) {}

DataGenerator::~DataGenerator() = default;

SortData DataGenerator::Generate(uint32_t size, uint32_t bits) {
  std::uniform_int_distribution<uint32_t> dist_values;
  std::uniform_int_distribution<uint32_t> dist_keys;
  if (bits < 32) dist_keys = std::uniform_int_distribution<uint32_t>(0, (1u << bits) - 1);

  SortData data;
  data.keys.reserve(size);
  data.values.reserve(size);
  for (int i = 0; i < size; ++i) {
    data.keys.push_back(dist_keys(gen_));
  }
  for (int i = 0; i < size; ++i) {
    data.values.push_back(dist_values(gen_));
  }
  return data;
}
