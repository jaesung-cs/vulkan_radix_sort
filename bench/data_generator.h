#ifndef VK_RADIX_SORT_DATA_GENERATOR_H
#define VK_RADIX_SORT_DATA_GENERATOR_H

#include <vector>
#include <random>

struct SortData {
  std::vector<uint32_t> keys;
  std::vector<uint32_t> values;
};

class DataGenerator {
 public:
  DataGenerator();
  
  explicit DataGenerator(int seed);

  ~DataGenerator();

  SortData Generate(uint32_t size, uint32_t bits = 32);

 private:
  std::mt19937 gen_;
};

#endif  // VK_RADIX_SORT_DATA_GENERATOR_H
