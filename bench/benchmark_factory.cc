#include "benchmark_factory.h"

#include "cpu_benchmark.h"
#include "vrdx_benchmark.h"

#ifdef CUDA
#include "cuda_benchmark.h"
#endif

std::unique_ptr<BenchmarkBase> BenchmarkFactory::Create(
    const std::string& type) {
  if (type == "cpu") return std::make_unique<CpuBenchmark>();
  if (type == "vrdx") return std::make_unique<VrdxBenchmark>();

#ifdef CUDA
  if (type == "cuda") return std::make_unique<CudaBenchmark>();
#endif

  throw std::runtime_error("Unavailable benchmark type: " + type);
}