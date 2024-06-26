#include "cpu_benchmark_base.h"

#include <algorithm>

namespace {

uint32_t bitfieldExtract(uint32_t value, int offset, int bits) {
  return (value >> offset) & ((1u << bits) - 1);
}

}  // namespace

CpuBenchmarkBase::CpuBenchmarkBase() = default;

CpuBenchmarkBase::~CpuBenchmarkBase() = default;

CpuBenchmarkBase::IntermediateResults CpuBenchmarkBase::Sort(
    const std::vector<uint32_t>& keys) {
  IntermediateResults result;
  result.keys[3] = keys;
  std::sort(result.keys[3].begin(), result.keys[3].end());
  return result;
}

CpuBenchmarkBase::IntermediateResults CpuBenchmarkBase::SortKeyValue(
    const std::vector<uint32_t>& keys, const std::vector<uint32_t>& values) {
  std::vector<uint32_t> indices;
  for (int i = 0; i < keys.size(); ++i) {
    indices.push_back(i);
  }

  std::stable_sort(indices.begin(), indices.end(),
                   [&](int lhs, int rhs) { return keys[lhs] < keys[rhs]; });

  IntermediateResults result;
  result.keys[3].reserve(keys.size());
  result.values.reserve(keys.size());
  for (int i = 0; i < keys.size(); ++i) {
    result.keys[3].push_back(keys[indices[i]]);
    result.values.push_back(values[indices[i]]);
  }
  return result;
}
