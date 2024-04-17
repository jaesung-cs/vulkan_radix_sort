#include "cpu_benchmark_base.h"

#include <algorithm>

namespace {

uint32_t bitfieldExtract(uint32_t value, int offset, int bits) {
  return (value >> offset) & ((1u << bits) - 1);
}

}  // namespace

CpuBenchmarkBase::CpuBenchmarkBase() = default;

CpuBenchmarkBase::~CpuBenchmarkBase() = default;

CpuBenchmarkBase::IntermediateResults CpuBenchmarkBase::GlobalHistogram(
    const std::vector<uint32_t>& keys) {
  constexpr uint32_t RADIX = 256;

  IntermediateResults result;
  result.histogram.resize(RADIX * 4);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < keys.size(); j++) {
      uint32_t radix = bitfieldExtract(keys[j], 8 * i, 8);
      result.histogram[RADIX * i + radix]++;
    }

    uint32_t prefixSum = 0;
    for (int j = 0; j < RADIX; j++) {
      uint32_t value = result.histogram[RADIX * i + j];
      result.histogram[RADIX * i + j] = prefixSum;
      prefixSum += value;
    }
  }
  return result;
}

CpuBenchmarkBase::IntermediateResults CpuBenchmarkBase::Sort(
    const std::vector<uint32_t>& keys) {
  IntermediateResults result;
  result.keys[3] = keys;
  std::sort(result.keys[3].begin(), result.keys[3].end());
  return result;
}
