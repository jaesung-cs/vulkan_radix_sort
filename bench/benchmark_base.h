#ifndef VK_RADIX_SORT_BENCHMARK_BASE_H
#define VK_RADIX_SORT_BENCHMARK_BASE_H

#include <vector>

struct BenchmarkResults {
  std::vector<uint32_t> keys;
  std::vector<uint32_t> values;
  uint64_t total_time;  // ns
};

class BenchmarkBase {
 public:
  BenchmarkBase() = default;
  virtual ~BenchmarkBase() = default;

  virtual BenchmarkResults Sort(const std::vector<uint32_t>& keys) = 0;
  virtual BenchmarkResults SortKeyValue(
      const std::vector<uint32_t>& keys,
      const std::vector<uint32_t>& values) = 0;

 private:
};

#endif  // VK_RADIX_SORT_BENCHMARK_BASE_H
