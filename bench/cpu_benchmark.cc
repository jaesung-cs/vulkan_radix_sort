#include "cpu_benchmark.h"

#include <algorithm>
#include <chrono>
#include <iostream>

namespace {

int64_t GetTimestamp() {
  return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

}  // namespace

CpuBenchmark::CpuBenchmark() = default;

CpuBenchmark::~CpuBenchmark() = default;

CpuBenchmark::Results CpuBenchmark::Sort(const std::vector<uint32_t>& keys) {
  Results result;
  result.keys = keys;
  auto start = GetTimestamp();
  std::sort(result.keys.begin(), result.keys.end());
  auto end = GetTimestamp();
  result.total_time = end - start;
  return result;
}

CpuBenchmark::Results CpuBenchmark::SortKeyValue(const std::vector<uint32_t>& keys,
                                                 const std::vector<uint32_t>& values) {
  std::vector<uint32_t> indices;
  auto N = keys.size();
  for (int i = 0; i < N; ++i) {
    indices.push_back(i);
  }

  auto start = GetTimestamp();
  std::stable_sort(indices.begin(), indices.end(),
                   [&](int lhs, int rhs) { return keys[lhs] < keys[rhs]; });
  auto end = GetTimestamp();

  Results result;
  result.keys.reserve(N);
  result.values.reserve(N);
  for (int i = 0; i < N; ++i) {
    result.keys.push_back(keys[indices[i]]);
    result.values.push_back(values[indices[i]]);
  }
  result.total_time = end - start;
  return result;
}
