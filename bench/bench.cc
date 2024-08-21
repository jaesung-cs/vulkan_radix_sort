#include <iostream>
#include <iomanip>
#include <string>

#include "data_generator.h"
#include "cpu_benchmark.h"
#include "benchmark_base.h"
#include "benchmark_factory.h"

void VerifyKeys(const SortData& data, const BenchmarkResults& result) {
  CpuBenchmark benchmark;
  auto answer = benchmark.Sort(data.keys);

  bool ok = true;
  int key_position = -1;
  for (int i = 0; i < answer.keys.size(); ++i) {
    if (answer.keys[i] != result.keys[i]) {
      ok = false;
      key_position = i;
      break;
    }
  }

  if (!ok) {
    std::cout << "Not OK" << std::endl;
    std::cout << "key at position: " << key_position << std::endl;

    for (int i = std::max(0, key_position - 5);
         i < std::min<int>(data.keys.size(), key_position + 5); ++i) {
      std::cout << answer.keys[i] << ' ';
    }
    std::cout << std::endl;
    for (int i = std::max(0, key_position - 5);
         i < std::min<int>(data.keys.size(), key_position + 5); ++i) {
      std::cout << result.keys[i] << ' ';
    }
    std::cout << std::endl;
  }
}

void VerifyKeyValues(const SortData& data, const BenchmarkResults& result) {
  CpuBenchmark benchmark;
  auto answer = benchmark.SortKeyValue(data.keys, data.values);

  bool ok = true;
  int key_position = -1;
  int value_position = -1;
  for (int i = 0; i < answer.keys.size(); ++i) {
    if (answer.keys[i] != result.keys[i]) {
      ok = false;
      key_position = i;
      break;
    }
  }
  for (int i = 0; i < answer.values.size(); ++i) {
    if (answer.values[i] != result.values[i]) {
      ok = false;
      value_position = i;
      break;
    }
  }

  if (!ok) {
    std::cout << "Not OK" << std::endl;
    if (key_position >= 0) {
      std::cout << "key at position: " << key_position << std::endl;

      for (int i = std::max(0, key_position - 5);
           i < std::min<int>(data.keys.size() - 1, key_position + 5); ++i) {
        std::cout << answer.keys[i] << ' ';
      }
      std::cout << std::endl;
      for (int i = std::max(0, key_position - 5);
           i < std::min<int>(data.keys.size() - 1, key_position + 5); ++i) {
        std::cout << result.keys[i] << ' ';
      }
      std::cout << std::endl;
    }
    if (value_position >= 0)
      std::cout << "value at position: " << value_position << std::endl;
  }
}

int main(int argc, char** argv) {
  std::cout << "vk_radix_sort benchmark" << std::endl;

  if (argc != 3 && argc != 4) {
    std::cout << "Usage: bench <N> <type> [--verify]" << std::endl;
    return 1;
  }

  // target: 15 GItems/s for key, 11 GItems/s for kv sort, 4.19e6 items (A100)
  int size = std::stoi(argv[1]);
  std::string type = argv[2];

  // TODO: argparse
  bool verify = false;
  if (argc == 4 && std::string(argv[3]) == "--verify") verify = true;

  auto benchmark = BenchmarkFactory::Create(type);
  // TODO: provide seed
  DataGenerator data_generator;

  {
    std::cout << "================ sort key value ================"
              << std::endl;
    auto data = data_generator.Generate(size);
    auto result = benchmark->SortKeyValue(data.keys, data.values);
    double perf = (static_cast<double>(size) / 1e9) /
                  (static_cast<double>(result.total_time) / 1e9);
    std::cout << "total time: " << static_cast<double>(result.total_time) / 1e6
              << "ms (" << perf << " GItems/s)" << std::endl;

    if (verify) {
      VerifyKeyValues(data, result);
    }
  }

  {
    std::cout << "================ sort ================" << std::endl;
    auto data = data_generator.Generate(size);

    auto result = benchmark->Sort(data.keys);
    double perf = (static_cast<double>(size) / 1e9) /
                  (static_cast<double>(result.total_time) / 1e9);
    std::cout << "total time: " << static_cast<double>(result.total_time) / 1e6
              << "ms (" << perf << " GItems/s)" << std::endl;

    if (verify) {
      VerifyKeys(data, result);
    }
  }

  {
    std::cout << "================ sort key value speed ================"
              << std::endl;

    for (int i = 0; i < 100; ++i) {
      auto data = data_generator.Generate(size);
      auto result = benchmark->SortKeyValue(data.keys, data.values);

      double perf = (static_cast<double>(size) / 1e9) /
                    (static_cast<double>(result.total_time) / 1e9);
      std::cout << "[" << i << "] total time: "
                << static_cast<double>(result.total_time) / 1e6 << "ms ("
                << perf << " GItems/s)" << std::endl;

      if (verify) {
        VerifyKeyValues(data, result);
      }
    }
  }

  return 0;
}
