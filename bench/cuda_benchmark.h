#ifndef VK_RADIX_SORT_CUDA_BENCHMARK_H
#define VK_RADIX_SORT_CUDA_BENCHMARK_H

#include "benchmark_base.h"

#include <cuda_runtime.h>

class CudaBenchmark : public BenchmarkBase {
 private:
  static constexpr uint32_t MAX_ELEMENT_COUNT = 1 << 25;

 public:
  CudaBenchmark();
  ~CudaBenchmark() override;

  BenchmarkResults Sort(const std::vector<uint32_t>& keys) override;
  BenchmarkResults SortKeyValue(const std::vector<uint32_t>& keys,
                                const std::vector<uint32_t>& values) override;

 private:
  cudaStream_t stream_ = 0;
  cudaEvent_t start_timestamp_ = nullptr;
  cudaEvent_t end_timestamp_ = nullptr;

  void* temp_storage_ = nullptr;
  size_t temp_storage_bytes_ = 0;
  void* keys_ptr_ = nullptr;
  void* values_ptr_ = nullptr;
  void* out_keys_ptr_ = nullptr;
  void* out_values_ptr_ = nullptr;
};

#endif  // VK_RADIX_SORT_CUDA_BENCHMARK_H
