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

  try {
    auto benchmark = BenchmarkFactory::Create(type);
    auto cpu_benchmark = BenchmarkFactory::Create("cpu");
    // TODO: provide seed
    DataGenerator data_generator;

    {
      std::cout << "================ sort ================" << std::endl;
      auto data = data_generator.Generate(size);

      auto result0 = benchmark->Sort(data.keys);
      double perf =
          (static_cast<double>(size) / 1e9) / (static_cast<double>(result0.total_time) / 1e9);
      std::cout << "total time: " << static_cast<double>(result0.total_time) / 1e6 << "ms (" << perf
                << " GItems/s)" << std::endl;

      auto result1 = cpu_benchmark->Sort(data.keys);
      for (int i = 0; i < data.keys.size(); ++i) {
        if (result0.keys[i] != result1.keys[i]) {
          std::cout << "wrong key at index " << i << std::endl;

          int i0 = std::max(i - 5, 0);
          int i1 = std::min<int>(i + 6, data.keys.size());
          std::cout << "keys   (" << std::setw(6) << type << "): ";
          for (int j = i0; j < i1; ++j) {
            std::cout << std::setw(9) << std::hex << data.keys[j];
          }
          std::cout << std::endl;
          std::cout << "keys   (answer): ";
          for (int j = i0; j < i1; ++j) {
            std::cout << std::setw(9) << std::hex << result1.keys[j];
          }
          std::cout << std::endl;
          break;
        }
      }
    }

    {
      std::cout << "================ sort key value ================" << std::endl;
      auto data = data_generator.Generate(size);
      auto result0 = benchmark->SortKeyValue(data.keys, data.values);

      double perf =
          (static_cast<double>(size) / 1e9) / (static_cast<double>(result0.total_time) / 1e9);
      std::cout << "total time: " << static_cast<double>(result0.total_time) / 1e6 << "ms (" << perf
                << " GItems/s)" << std::endl;

      auto result1 = cpu_benchmark->SortKeyValue(data.keys, data.values);
      for (int i = 0; i < data.keys.size(); ++i) {
        if (result0.keys[i] != result1.keys[i]) {
          std::cout << "wrong key at index " << i << std::endl;

          int i0 = std::max(i - 5, 0);
          int i1 = std::min<int>(i + 6, data.keys.size());
          std::cout << "keys   (" << std::setw(6) << type << "): ";
          for (int j = i0; j < i1; ++j) {
            std::cout << std::setw(9) << std::hex << result0.keys[j];
          }
          std::cout << std::endl;
          std::cout << "keys   (answer): ";
          for (int j = i0; j < i1; ++j) {
            std::cout << std::setw(9) << std::hex << result1.keys[j];
          }
          std::cout << std::endl;
          std::cout << "values (" << std::setw(6) << type << "): ";
          for (int j = i0; j < i1; ++j) {
            std::cout << std::setw(9) << std::hex << result0.values[j];
          }
          std::cout << std::endl;
          std::cout << "values (answer): ";
          for (int j = i0; j < i1; ++j) {
            std::cout << std::setw(9) << std::hex << result1.values[j];
          }
          std::cout << std::endl;
          break;
        }
      }

      for (int i = 0; i < data.keys.size(); ++i) {
        if (result0.values[i] != result1.values[i]) {
          std::cout << "wrong value at index " << i << std::endl;

          int i0 = std::max(i - 5, 0);
          int i1 = std::min<int>(i + 6, data.keys.size());
          std::cout << "keys   (" << std::setw(6) << type << "): ";
          for (int j = i0; j < i1; ++j) {
            std::cout << std::setw(9) << std::hex << result0.keys[j];
          }
          std::cout << std::endl;
          std::cout << "keys   (answer): ";
          for (int j = i0; j < i1; ++j) {
            std::cout << std::setw(9) << std::hex << result1.keys[j];
          }
          std::cout << std::endl;
          std::cout << "values (" << std::setw(6) << type << "): ";
          for (int j = i0; j < i1; ++j) {
            std::cout << std::setw(9) << std::hex << result0.values[j];
          }
          std::cout << std::endl;
          std::cout << "values (answer): ";
          for (int j = i0; j < i1; ++j) {
            std::cout << std::setw(9) << std::hex << result1.values[j];
          }
          std::cout << std::endl;
          std::cout << std::endl;
          break;
        }
      }

      /*
      for (int i = 0; i < data.keys.size(); ++i) {
        std::cout << std::setw(8) << std::hex << result0.keys[i] << ' '
                  << std::setw(8) << std::hex << result0.values[i] << std::endl;
      }
      */
    }

    {
      std::cout << "================ sort key value speed ================" << std::endl;

      for (int i = 0; i < 20; ++i) {
        auto data = data_generator.Generate(size);
        auto result = benchmark->SortKeyValue(data.keys, data.values);

        double perf =
            (static_cast<double>(size) / 1e9) / (static_cast<double>(result.total_time) / 1e9);
        std::cout << "[" << i << "] total time: " << static_cast<double>(result.total_time) / 1e6
                  << "ms (" << perf << " GItems/s)" << std::endl;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
