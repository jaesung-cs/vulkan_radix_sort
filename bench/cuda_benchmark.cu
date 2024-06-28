#include "cuda_benchmark.h"

#include <cub/cub.cuh>

CudaBenchmark::CudaBenchmark() {
  cudaStreamCreate(&stream_);
  cudaEventCreate(&start_timestamp_);
  cudaEventCreate(&end_timestamp_);

  // allocate in/out memories
  cudaMalloc(&keys_ptr_, MAX_ELEMENT_COUNT * sizeof(uint32_t));
  cudaMalloc(&values_ptr_, MAX_ELEMENT_COUNT * sizeof(uint32_t));
  cudaMalloc(&out_keys_ptr_, MAX_ELEMENT_COUNT * sizeof(uint32_t));
  cudaMalloc(&out_values_ptr_, MAX_ELEMENT_COUNT * sizeof(uint32_t));
}

CudaBenchmark::~CudaBenchmark() {
  cudaStreamDestroy(stream_);
  cudaEventDestroy(start_timestamp_);
  cudaEventDestroy(end_timestamp_);

  if (temp_storage_) cudaFree(temp_storage_);
  cudaFree(keys_ptr_);
  cudaFree(values_ptr_);
  cudaFree(out_keys_ptr_);
  cudaFree(out_values_ptr_);
}

CudaBenchmark::Results CudaBenchmark::Sort(const std::vector<uint32_t> &keys) {
  auto n = keys.size();

  Results result;
  result.keys.resize(n);

  // CPU to GPU
  cudaMemcpy(keys_ptr_, keys.data(), n * sizeof(uint32_t), cudaMemcpyDefault);

  // allocate temp storage
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes,
                                 static_cast<const uint32_t *>(keys_ptr_),
                                 static_cast<uint32_t *>(out_keys_ptr_), n);
  if (temp_storage_bytes_ < temp_storage_bytes) {
    if (temp_storage_) cudaFree(temp_storage_);
    cudaMalloc(&temp_storage_, temp_storage_bytes);
    temp_storage_bytes_ = temp_storage_bytes;
  }

  // cub sort, measure time
  cudaStreamSynchronize(stream_);
  cudaEventRecord(start_timestamp_, stream_);
  cub::DeviceRadixSort::SortKeys(temp_storage_, temp_storage_bytes_,
                                 static_cast<const uint32_t *>(keys_ptr_),
                                 static_cast<uint32_t *>(out_keys_ptr_), n, 0,
                                 32, stream_);
  cudaEventRecord(end_timestamp_, stream_);
  cudaStreamSynchronize(stream_);

  // GPU to CPU
  cudaMemcpy(result.keys.data(), out_keys_ptr_, n * sizeof(uint32_t),
             cudaMemcpyDefault);

  // measure time
  float ms;
  cudaEventElapsedTime(&ms, start_timestamp_, end_timestamp_);
  result.total_time = static_cast<uint64_t>(ms * 1e6);  // ms to ns

  return result;
}

CudaBenchmark::Results CudaBenchmark::SortKeyValue(
    const std::vector<uint32_t> &keys, const std::vector<uint32_t> &values) {
  auto n = keys.size();

  Results result;
  result.keys.resize(n);
  result.values.resize(n);

  // CPU to GPU
  cudaMemcpy(keys_ptr_, keys.data(), n * sizeof(uint32_t), cudaMemcpyDefault);
  cudaMemcpy(values_ptr_, values.data(), n * sizeof(uint32_t),
             cudaMemcpyDefault);

  // allocate temp storage
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                  static_cast<const uint32_t *>(keys_ptr_),
                                  static_cast<uint32_t *>(out_keys_ptr_),
                                  static_cast<const uint32_t *>(values_ptr_),
                                  static_cast<uint32_t *>(out_values_ptr_), n);
  if (temp_storage_bytes_ < temp_storage_bytes) {
    if (temp_storage_) cudaFree(temp_storage_);
    cudaMalloc(&temp_storage_, temp_storage_bytes);
    temp_storage_bytes_ = temp_storage_bytes;
  }

  // cub sort, measure time
  cudaStreamSynchronize(stream_);
  cudaEventRecord(start_timestamp_, stream_);
  cub::DeviceRadixSort::SortPairs(temp_storage_, temp_storage_bytes,
                                  static_cast<const uint32_t *>(keys_ptr_),
                                  static_cast<uint32_t *>(out_keys_ptr_),
                                  static_cast<const uint32_t *>(values_ptr_),
                                  static_cast<uint32_t *>(out_values_ptr_), n,
                                  0, 32, stream_);
  cudaEventRecord(end_timestamp_, stream_);
  cudaStreamSynchronize(stream_);

  // GPU to CPU
  cudaMemcpy(result.keys.data(), out_keys_ptr_, n * sizeof(uint32_t),
             cudaMemcpyDefault);
  cudaMemcpy(result.values.data(), out_values_ptr_, n * sizeof(uint32_t),
             cudaMemcpyDefault);

  // measure time
  float ms;
  cudaEventElapsedTime(&ms, start_timestamp_, end_timestamp_);
  result.total_time = static_cast<uint64_t>(ms * 1e6);  // ms to ns
  return result;
}
