cmake_minimum_required(VERSION 3.15)

project(vk_radix_sort LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Vulkan REQUIRED)

# find slangc
string(REPLACE glslangValidator slangc SLANGC_EXECUTABLE ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE})
message("slangc executable: ${SLANGC_EXECUTABLE}")

# adds -fPIC, works for linux, when building shared library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# shaders
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/generated)

# build_shader(TARGET SHADER OUTPUT DEFINE...)
function(build_shader)
  list(POP_FRONT ARGV TARGET SHADER OUTPUT)
  list(TRANSFORM ARGV PREPEND "-D" OUTPUT_VARIABLE DEFINES)

  get_filename_component(SHADER ${SHADER} ABSOLUTE)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h
    COMMAND
      python ${CMAKE_CURRENT_SOURCE_DIR}/tools/slangc_to_header.py
      -target spirv
      -fvk-b-shift 0 0
      -o ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h
      ${DEFINES}
      ${SHADER}
    DEPENDS ${SHADER}
    COMMENT "Compiling ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h"
  )

  add_custom_target(${OUTPUT} DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h)
  add_dependencies(${TARGET} ${OUTPUT})
endfunction()

# library
add_library(vk_radix_sort STATIC
  src/vk_radix_sort.cc
)

target_include_directories(vk_radix_sort
  PUBLIC include
  PRIVATE src
)

target_link_libraries(vk_radix_sort
  PUBLIC Vulkan::Vulkan
)

build_shader(vk_radix_sort src/shader/upsweep.slang upsweep_slang)
build_shader(vk_radix_sort src/shader/spine.slang spine_slang)
build_shader(vk_radix_sort src/shader/downsweep.slang downsweep_slang)
build_shader(vk_radix_sort src/shader/downsweep.slang downsweep_key_value_slang KEY_VALUE)

# bench
if (PROJECT_IS_TOP_LEVEL)
  set(BENCH_SOURCES
    bench/bench.cc
    bench/benchmark_factory.cc
    bench/cpu_benchmark.cc
    bench/data_generator.cc
    bench/vma_impl.cc
    bench/vulkan_benchmark.cc
  )

  # if CUDA is available, add CUB benchmark
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
    list(APPEND BENCH_SOURCES
      bench/cuda_benchmark.cu
    )
  endif()

  add_executable(bench ${BENCH_SOURCES})

  if (CMAKE_CUDA_COMPILER)
    target_compile_definitions(bench PUBLIC BENCH_CUDA)
  endif()

  # if VMA is already added from parent project, skip
  if (NOT TARGET VulkanMemoryAllocator)
    add_subdirectory(third_party/VulkanMemoryAllocator EXCLUDE_FROM_ALL)
  endif()

  target_link_libraries(bench PRIVATE vk_radix_sort VulkanMemoryAllocator)
endif()
