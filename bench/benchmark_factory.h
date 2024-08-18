#ifndef VK_RADIX_SORT_BENCHMARK_FACTORY_H
#define VK_RADIX_SORT_BENCHMARK_FACTORY_H

#include <memory>
#include <string>

class BenchmarkBase;

class BenchmarkFactory {
 public:
  static std::unique_ptr<BenchmarkBase> Create(const std::string& type);
};

#endif  // VK_RADIX_SORT_BENCHMARK_FACTORY_H
