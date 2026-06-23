#ifndef VK_RADIX_SORT_BENCHMARK_BASE_H
#define VK_RADIX_SORT_BENCHMARK_BASE_H

#include <string>
#include <vector>
#include <cstdint>

class BenchmarkBase {
 public:
  struct Results {
    std::vector<uint32_t> keys;
    std::vector<uint32_t> values;
    uint64_t total_time = 0;    // ns (GPU timestamps, total sort time)
    uint64_t cpu_time = 0;      // ns (wall clock, submit to fence signaled)
    uint64_t upsweep_ns = 0;    // ns, summed across 4 passes
    uint64_t spine_ns = 0;      // ns, summed across 4 passes
    uint64_t downsweep_ns = 0;  // ns, summed across 4 passes
  };

 public:
  BenchmarkBase() = default;
  virtual ~BenchmarkBase() = default;

  virtual std::string LibraryVersion() const { return ""; }

  virtual Results Sort(const std::vector<uint32_t>& keys) = 0;
  virtual Results SortKeyValue(const std::vector<uint32_t>& keys,
                               const std::vector<uint32_t>& values) = 0;
};

#endif  // VK_RADIX_SORT_BENCHMARK_BASE_H
