
#include <random>

std::vector<uint32_t> GenerateUniformRandomData(uint32_t size, uint32_t bits) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dist;
  if (bits < 32)
    dist = std::uniform_int_distribution<uint32_t>(0, (1u << bits) - 1);

  std::vector<uint32_t> data;
  data.reserve(size);
  for (int i = 0; i < size; i++) {
    data.push_back(dist(gen));
  }
  return data;
}
