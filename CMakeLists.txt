cmake_minimum_required(VERSION 3.15)

project(vk_radix_sort LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Vulkan REQUIRED)

include(FetchContent)
FetchContent_Declare(
  volk
  GIT_REPOSITORY https://github.com/zeux/volk.git
  GIT_TAG        f30088b3f4160810b53e19258dd2f7395e5f0ba3  # vulkan-sdk-1.4.328.1
)
FetchContent_MakeAvailable(volk)

# find slangc
string(REPLACE glslangValidator slangc SLANGC_EXECUTABLE ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE})
message("slangc executable: ${SLANGC_EXECUTABLE}")

# shaders
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/generated)

# build_shader(SHADER OUTPUT DEFINE...)
function(build_shader)
  list(POP_FRONT ARGV SHADER OUTPUT)
  list(TRANSFORM ARGV PREPEND "-D" OUTPUT_VARIABLE DEFINES)

  get_filename_component(SHADER ${SHADER} ABSOLUTE)

  add_custom_target(${OUTPUT} ALL
    COMMAND
      python ${CMAKE_CURRENT_SOURCE_DIR}/tools/slangc_to_header.py
      -target spirv
      -fvk-t-shift 0 0
      -fvk-u-shift 0 0
      -o ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h
      ${DEFINES}
      ${SHADER}
    COMMENT "Compiling ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h"
  )
endfunction()

# header-only library
add_library(vk_radix_sort INTERFACE)

target_include_directories(vk_radix_sort INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

build_shader(src/shader/upsweep.slang upsweep_slang)
build_shader(src/shader/spine.slang spine_slang)
build_shader(src/shader/downsweep.slang downsweep_slang)
build_shader(src/shader/downsweep.slang downsweep_key_value_slang KEY_VALUE)

# bench
if (PROJECT_IS_TOP_LEVEL)
  include(FetchContent)
  FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG        f30088b3f4160810b53e19258dd2f7395e5f0ba3  # vulkan-sdk-1.4.328.1
  )
  FetchContent_MakeAvailable(volk)

  set(BENCH_SOURCES
    bench/bench.cc
    bench/benchmark_factory.cc
    bench/cpu_benchmark.cc
    bench/data_generator.cc
    bench/vma_impl.cc
    bench/volk_impl.cc
    bench/vrdx_impl.cc
    bench/vulkan_benchmark.cc
  )

  # if CUDA is available, add CUB benchmark
  # include(CheckLanguage)
  # check_language(CUDA)
  # if (CMAKE_CUDA_COMPILER)
  #   enable_language(CUDA)
  #   set(CMAKE_CUDA_STANDARD 17)
  #   set(CMAKE_CUDA_STANDARD_REQUIRED True)
  #   list(APPEND BENCH_SOURCES
  #     bench/cuda_benchmark.cu
  #   )
  # endif()

  add_executable(bench ${BENCH_SOURCES})

  # if (CMAKE_CUDA_COMPILER)
  #   target_compile_definitions(bench PUBLIC BENCH_CUDA)
  # endif()

  # if VMA is already added from parent project, skip
  if (NOT TARGET VulkanMemoryAllocator)
    add_subdirectory(third_party/VulkanMemoryAllocator EXCLUDE_FROM_ALL)
  endif()

  target_compile_definitions(bench PRIVATE VK_NO_PROTOTYPES)

  target_link_libraries(bench PRIVATE vk_radix_sort volk::volk_headers VulkanMemoryAllocator)
endif()
