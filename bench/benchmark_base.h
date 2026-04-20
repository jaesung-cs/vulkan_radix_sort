#ifndef VK_RADIX_SORT_BENCHMARK_BASE_H
#define VK_RADIX_SORT_BENCHMARK_BASE_H

#include <vector>
#include <cstdint>

class BenchmarkBase {
 public:
  struct Results {
    std::vector<uint32_t> keys;
    std::vector<uint32_t> values;
    uint64_t total_time = 0;  // ns (GPU timestamps)
    uint64_t cpu_time = 0;    // ns (wall clock, submit to fence signaled)
  };

 public:
  BenchmarkBase() = default;
  virtual ~BenchmarkBase() = default;

  virtual Results Sort(const std::vector<uint32_t>& keys) = 0;
  virtual Results SortKeyValue(const std::vector<uint32_t>& keys,
                               const std::vector<uint32_t>& values) = 0;

};

#endif  // VK_RADIX_SORT_BENCHMARK_BASE_H
