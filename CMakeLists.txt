cmake_minimum_required(VERSION 3.21)

project(vk_radix_sort LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Vulkan REQUIRED COMPONENTS shaderc_combined)

# if VMA is already added from parent project, skip
if (NOT TARGET VulkanMemoryAllocator)
  add_subdirectory(third_party/VulkanMemoryAllocator EXCLUDE_FROM_ALL)
endif()

# adds -fPIC, works for linux, when building shared library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(vk_radix_sort STATIC
  src/vk_radix_sort.cc
)

target_include_directories(vk_radix_sort
  PUBLIC include
  PRIVATE src
)

target_link_libraries(vk_radix_sort
  PUBLIC Vulkan::Vulkan VulkanMemoryAllocator
  PRIVATE Vulkan::shaderc_combined
)

# bench
if (PROJECT_IS_TOP_LEVEL)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)

  add_executable(bench
    bench/bench.cc
    bench/cpu_benchmark_base.cc
    bench/cuda_benchmark_base.cu
    bench/data_generator.cc
    bench/vma_impl.cc
    bench/vulkan_benchmark_base.cc
  )
  target_link_libraries(bench PRIVATE vk_radix_sort)
endif()
