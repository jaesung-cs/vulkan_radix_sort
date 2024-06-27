#ifndef VK_RADIX_SORT_CUDA_BENCHMARK_BASE_H
#define VK_RADIX_SORT_CUDA_BENCHMARK_BASE_H

#include <vector>

#include <cuda_runtime.h>

class CudaBenchmarkBase {
 private:
  struct Results {
    std::vector<uint32_t> keys;
    std::vector<uint32_t> values;
    uint64_t total_time = 0;  // ns
  };

  static constexpr uint32_t MAX_ELEMENT_COUNT = 1 << 25;

 public:
  CudaBenchmarkBase();
  ~CudaBenchmarkBase();

  Results Sort(const std::vector<uint32_t>& keys);
  Results SortKeyValue(const std::vector<uint32_t>& keys,
                       const std::vector<uint32_t>& values);

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

#endif  // VK_RADIX_SORT_CUDA_BENCHMARK_BASE_H
