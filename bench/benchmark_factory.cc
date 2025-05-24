#include "benchmark_factory.h"

#include <stdexcept>

#include "cpu_benchmark.h"
#include "vulkan_benchmark.h"

#ifdef BENCH_CUDA
#include "cuda_benchmark.h"
#endif

std::unique_ptr<BenchmarkBase> BenchmarkFactory::Create(const std::string& type) {
  if (type == "cpu") return std::make_unique<CpuBenchmark>();
  if (type == "vulkan") return std::make_unique<VulkanBenchmark>();

#ifdef BENCH_CUDA
  if (type == "cuda") return std::make_unique<CudaBenchmark>();
#endif

  throw std::runtime_error("Unavailable benchmark type: " + type);
}