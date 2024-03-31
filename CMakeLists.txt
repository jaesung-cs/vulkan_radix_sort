cmake_minimum_required(VERSION 3.24)

project(vk_radix_sort LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Vulkan REQUIRED COMPONENTS shaderc_combined)
add_subdirectory(third_party/VulkanMemoryAllocator EXCLUDE_FROM_ALL)

add_library(vk_radix_sort STATIC
  src/vk_radix_sort.cc
)

target_include_directories(vk_radix_sort
  PUBLIC include
  PRIVATE src
)

target_link_libraries(vk_radix_sort
  PUBLIC Vulkan::Vulkan
  PRIVATE Vulkan::shaderc_combined
)

# bench
add_executable(bench bench/bench.cc bench/vma_impl.cc)
target_link_libraries(bench PRIVATE vk_radix_sort VulkanMemoryAllocator)
