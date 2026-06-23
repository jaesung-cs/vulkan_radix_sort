#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cxxopts.hpp>

#include "benchmark_factory.h"
#include "benchmark_base.h"
#include "data_generator.h"

namespace {

constexpr int kWarmupRuns = 1;
constexpr int kTimedRuns = 10;
constexpr uint32_t kNMin = 1u << 18;
constexpr uint32_t kNMax = 1u << 25;
constexpr int kNCount = 128;
constexpr uint32_t kNStep = (kNMax - kNMin) / (kNCount - 1);

double toMs(uint64_t ns) { return static_cast<double>(ns) / 1e6; }
double toGItemsS(uint32_t n, uint64_t ns) {
  return (static_cast<double>(n) / 1e9) / (static_cast<double>(ns) / 1e9);
}

uint64_t median(std::vector<uint64_t>& v) {
  auto mid = static_cast<std::ptrdiff_t>(v.size() / 2);
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  return v[static_cast<size_t>(mid)];
}

struct Row {
  uint32_t n;
  std::string sort;
  double gpu_ms, cpu_ms;
  double gpu_gitems_s, cpu_gitems_s;
  double upsweep_ms, spine_ms, downsweep_ms;  // summed across 4 passes; 0 for non-Vulkan
};

bool checkCorrectness(BenchmarkBase* bench, BenchmarkBase* cpu, uint32_t n, DataGenerator& gen) {
  auto data = gen.Generate(n);

  auto r0 = bench->Sort(data.keys);
  auto r1 = cpu->Sort(data.keys);
  for (uint32_t i = 0; i < n; ++i) {
    if (r0.keys[i] != r1.keys[i]) {
      std::cerr << "Sort correctness failed at index " << i << std::endl;
      return false;
    }
  }

  auto r2 = bench->SortKeyValue(data.keys, data.values);
  auto r3 = cpu->SortKeyValue(data.keys, data.values);
  for (uint32_t i = 0; i < n; ++i) {
    if (r2.keys[i] != r3.keys[i] || r2.values[i] != r3.values[i]) {
      std::cerr << "SortKeyValue correctness failed at index " << i << std::endl;
      return false;
    }
  }

  std::cout << "Correctness check passed (N=" << n << ")" << std::endl;
  return true;
}

Row measure(BenchmarkBase* bench, uint32_t n, const std::string& sort, DataGenerator& gen) {
  // warmup
  for (int i = 0; i < kWarmupRuns; ++i) {
    auto data = gen.Generate(n);
    if (sort == "keys")
      bench->Sort(data.keys);
    else
      bench->SortKeyValue(data.keys, data.values);
  }

  std::vector<uint64_t> gpu_times, cpu_times, upsweep_times, spine_times, downsweep_times;
  gpu_times.reserve(kTimedRuns);
  cpu_times.reserve(kTimedRuns);
  upsweep_times.reserve(kTimedRuns);
  spine_times.reserve(kTimedRuns);
  downsweep_times.reserve(kTimedRuns);

  for (int i = 0; i < kTimedRuns; ++i) {
    auto data = gen.Generate(n);
    BenchmarkBase::Results r;
    if (sort == "keys")
      r = bench->Sort(data.keys);
    else
      r = bench->SortKeyValue(data.keys, data.values);
    gpu_times.push_back(r.total_time);
    cpu_times.push_back(r.cpu_time);
    upsweep_times.push_back(r.upsweep_ns);
    spine_times.push_back(r.spine_ns);
    downsweep_times.push_back(r.downsweep_ns);
  }

  uint64_t gpu_med = median(gpu_times);
  uint64_t cpu_med = median(cpu_times);
  uint64_t up_med = median(upsweep_times);
  uint64_t sp_med = median(spine_times);
  uint64_t dn_med = median(downsweep_times);

  return Row{n,
             sort,
             toMs(gpu_med),
             toMs(cpu_med),
             toGItemsS(n, gpu_med),
             toGItemsS(n, cpu_med),
             toMs(up_med),
             toMs(sp_med),
             toMs(dn_med)};
}

}  // namespace

int main(int argc, char** argv) {
  cxxopts::Options options("bench", "Vulkan radix sort benchmark");
  options.add_options()("type", "Backend type", cxxopts::value<std::string>())(
      "o,output", "Output CSV file", cxxopts::value<std::string>()->default_value("results.csv"))(
      "validation", "Enable Vulkan validation layers")(
      "no-verify", "Skip correctness check and proceed to benchmarking")("h,help", "Print usage");
  options.parse_positional({"type"});
  options.positional_help("<type>");
  options.custom_help(
      "[options]\n\nTypes:\n"
      "  vulkan    Vulkan (this library)\n"
      "  fuchsia   Fuchsia radix sort (Vulkan)\n"
      "  cuda      CUB Onesweep (CUDA)\n"
      "  cpu       std::sort reference\n");

  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (const cxxopts::exceptions::exception& e) {
    std::cerr << e.what() << "\n" << options.help() << std::endl;
    return 1;
  }

  if (result.count("help") || !result.count("type")) {
    std::cout << options.help() << std::endl;
    return result.count("help") ? 0 : 1;
  }

  std::string type = result["type"].as<std::string>();
  std::string csv_path = result["output"].as<std::string>();
  bool validation = result.count("validation") > 0;
  bool no_verify = result.count("no-verify") > 0;

  std::unique_ptr<BenchmarkBase> bench, cpu;
  try {
    bench = BenchmarkFactory::Create(type, validation);
    cpu = BenchmarkFactory::Create("cpu");
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  DataGenerator gen;
  std::vector<Row> rows;

  for (int i = 0; i < kNCount; ++i) {
    uint32_t n = kNMin + static_cast<uint32_t>(i) * kNStep;

    if (i == 0 && !no_verify) {
      if (!checkCorrectness(bench.get(), cpu.get(), n, gen)) return 1;
    }

    for (const std::string& sort : {"keys", "kv"}) {
      Row row = measure(bench.get(), n, sort, gen);
      rows.push_back(row);

      std::cout << "[" << std::setw(3) << i + 1 << "/" << kNCount << "]"
                << " N=" << std::setw(9) << n << " [" << std::setw(4) << sort << "]"
                << "  gpu: " << std::fixed << std::setprecision(3) << row.gpu_ms << "ms"
                << " (" << std::setprecision(2) << row.gpu_gitems_s << " GItems/s)"
                << "  cpu: " << std::setprecision(3) << row.cpu_ms << "ms"
                << " (" << std::setprecision(2) << row.cpu_gitems_s << " GItems/s)";
      if (row.upsweep_ms > 0 || row.spine_ms > 0 || row.downsweep_ms > 0) {
        auto pct = [&](double ms) -> int {
          return row.gpu_ms > 0 ? static_cast<int>(ms / row.gpu_ms * 100 + 0.5) : 0;
        };
        std::cout << std::fixed << std::setprecision(3) << "  [up=" << row.upsweep_ms << "ms("
                  << pct(row.upsweep_ms) << "%)"
                  << " sp=" << row.spine_ms << "ms(" << pct(row.spine_ms) << "%)"
                  << " dn=" << row.downsweep_ms << "ms(" << pct(row.downsweep_ms) << "%)" << "]";
      }
      std::cout << std::endl;
    }
  }

  std::ofstream csv(csv_path);
  if (!csv) {
    std::cerr << "Failed to open " << csv_path << " for writing" << std::endl;
    return 1;
  }

  std::string lib_ver = bench->LibraryVersion();
  if (!lib_ver.empty()) csv << "# version: " << lib_ver << "\n";
  csv << "backend,n,sort,gpu_ms,cpu_ms,gpu_gitems_s,cpu_gitems_s\n";
  for (const auto& r : rows) {
    csv << type << "," << r.n << "," << r.sort << "," << std::fixed << std::setprecision(6)
        << r.gpu_ms << "," << r.cpu_ms << "," << r.gpu_gitems_s << "," << r.cpu_gitems_s << "\n";
  }

  std::cout << "\nResults written to " << csv_path << std::endl;
  return 0;
}
