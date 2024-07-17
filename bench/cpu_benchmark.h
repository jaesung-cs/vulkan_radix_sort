#ifndef VK_RADIX_SORT_CPU_BENCHMARK_H
#define VK_RADIX_SORT_CPU_BENCHMARK_H

#include "benchmark_base.h"

class CpuBenchmark : public BenchmarkBase {
 public:
  CpuBenchmark();
  ~CpuBenchmark() override;

  Results Sort(const std::vector<uint32_t>& keys) override;
  Results SortKeyValue(const std::vector<uint32_t>& keys,
                       const std::vector<uint32_t>& values) override;

 private:
};

#endif  // VK_RADIX_SORT_CPU_BENCHMARK_H
