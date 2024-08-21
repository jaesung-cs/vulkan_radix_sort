#include "cpu_benchmark.h"

#include <algorithm>

CpuBenchmark::CpuBenchmark() = default;

CpuBenchmark::~CpuBenchmark() = default;

BenchmarkResults CpuBenchmark::Sort(const std::vector<uint32_t>& keys) {
  BenchmarkResults result;
  result.keys = keys;
  std::sort(result.keys.begin(), result.keys.end());
  return result;
}

BenchmarkResults CpuBenchmark::SortKeyValue(
    const std::vector<uint32_t>& keys, const std::vector<uint32_t>& values) {
  std::vector<uint32_t> indices;
  auto N = keys.size();
  for (int i = 0; i < N; ++i) {
    indices.push_back(i);
  }

  std::stable_sort(indices.begin(), indices.end(),
                   [&](int lhs, int rhs) { return keys[lhs] < keys[rhs]; });

  BenchmarkResults result;
  result.keys.reserve(N);
  result.values.reserve(N);
  for (int i = 0; i < N; ++i) {
    result.keys.push_back(keys[indices[i]]);
    result.values.push_back(values[indices[i]]);
  }
  return result;
}
