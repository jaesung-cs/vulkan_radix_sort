#include <iostream>
#include <iomanip>
#include <string>

#include "data_generator.h"
#include "benchmark_base.h"
#include "benchmark_factory.h"

int main(int argc, char** argv) {
  std::cout << "vk_radix_sort benchmark" << std::endl;

  if (argc != 3) {
    std::cout << "Usage: bench <N> <type>" << std::endl;
    return 1;
  }

  // target: 15 GItems/s for key, 11 GItems/s for kv sort, 4.19e6 items (A100)
  int size = std::stoi(argv[1]);
  std::string type = argv[2];

  auto benchmark = BenchmarkFactory::Create(type);
  // TODO: provide seed
  DataGenerator data_generator;

  {
    std::cout << "================ sort ================" << std::endl;
    auto data = data_generator.Generate(size);

    auto result0 = benchmark->Sort(data.keys);
    {
      double perf = (static_cast<double>(size) / 1e9) /
                    (static_cast<double>(result0.total_time) / 1e9);
      std::cout << "total time: "
                << static_cast<double>(result0.total_time) / 1e6 << "ms ("
                << perf << " GItems/s)" << std::endl;
    }
  }

  {
    std::cout << "================ sort key value ================"
              << std::endl;
    auto data = data_generator.Generate(size);
    auto result0 = benchmark->SortKeyValue(data.keys, data.values);
    {
      double perf = (static_cast<double>(size) / 1e9) /
                    (static_cast<double>(result0.total_time) / 1e9);
      std::cout << "total time: "
                << static_cast<double>(result0.total_time) / 1e6 << "ms ("
                << perf << " GItems/s)" << std::endl;
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
    }
  }

  return 0;
}
