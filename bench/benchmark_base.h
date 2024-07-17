#ifndef VK_RADIX_SORT_BENCHMARK_BASE_H
#define VK_RADIX_SORT_BENCHMARK_BASE_H

#include <vector>

class BenchmarkBase {
 protected:
  struct Results {
    std::vector<uint32_t> keys;
    std::vector<uint32_t> values;
    uint64_t total_time;  // ns
  };

 public:
  BenchmarkBase() = default;
  virtual ~BenchmarkBase() = default;

  virtual Results Sort(const std::vector<uint32_t>& keys) = 0;
  virtual Results SortKeyValue(const std::vector<uint32_t>& keys,
                               const std::vector<uint32_t>& values) = 0;

 private:
};

#endif  // VK_RADIX_SORT_BENCHMARK_BASE_H
