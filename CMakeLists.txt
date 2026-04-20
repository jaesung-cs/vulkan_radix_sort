cmake_minimum_required(VERSION 3.24)

project(vk_radix_sort VERSION 0.3.0 LANGUAGES CXX)

find_package(Vulkan REQUIRED)
find_package(Python3 REQUIRED)
include(FetchContent)

# find slangc
string(REPLACE glslangValidator slangc SLANGC_EXECUTABLE ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE})
message(STATUS "slangc executable: ${SLANGC_EXECUTABLE}")

# shaders
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/generated)

# build_shader(SHADER OUTPUT DEFINE...)
function(build_shader)
  list(POP_FRONT ARGV SHADER OUTPUT)
  list(TRANSFORM ARGV PREPEND "-D" OUTPUT_VARIABLE DEFINES)

  get_filename_component(SHADER ${SHADER} ABSOLUTE)

  add_custom_target(${OUTPUT} ALL
    COMMAND
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/slangc_to_header.py
      -target spirv
      -fvk-t-shift 0 0
      -fvk-u-shift 0 0
      -o ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h
      ${DEFINES}
      ${SHADER}
    COMMENT "Compiling ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h"
    VERBATIM
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

add_custom_target(vk_radix_sort_header ALL
  COMMAND
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_header.py
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vk_radix_sort.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vk_radix_sort.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/generated
    ${PROJECT_VERSION_MAJOR} ${PROJECT_VERSION_MINOR} ${PROJECT_VERSION_PATCH}
  DEPENDS
    upsweep_slang
    spine_slang
    downsweep_slang
    downsweep_key_value_slang
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vk_radix_sort.h.in
  COMMENT "Generating include/vk_radix_sort.h from template"
  VERBATIM
)

add_dependencies(vk_radix_sort vk_radix_sort_header)

# bench
if(PROJECT_IS_TOP_LEVEL)
  FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG        f30088b3f4160810b53e19258dd2f7395e5f0ba3  # vulkan-sdk-1.4.328.1
  )

  FetchContent_Declare(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        1d8f600fd424278486eade7ed3e877c99f0846b1  # v3.3.0
  )

  FetchContent_MakeAvailable(volk VulkanMemoryAllocator)

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

  #if CUDA is available, add CUB benchmark
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    list(APPEND BENCH_SOURCES
      bench/cuda_benchmark.cu
    )
  endif()

  add_executable(bench ${BENCH_SOURCES})

  set_target_properties(bench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
  )

  if(CMAKE_CUDA_COMPILER)
    set_target_properties(bench PROPERTIES CUDA_ARCHITECTURES native)
    if(MSVC)
      target_compile_options(bench PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor /utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor -Xcompiler=/utf-8>
      )
      target_link_options(bench PRIVATE
        $<$<CONFIG:Debug>:/NODEFAULTLIB:LIBCMTD>
        $<$<NOT:$<CONFIG:Debug>>:/NODEFAULTLIB:LIBCMT>
      )
    endif()
    target_compile_definitions(bench PUBLIC BENCH_CUDA)
  endif()

  target_compile_definitions(bench PRIVATE VK_NO_PROTOTYPES VRDX_USE_VOLK)

  target_link_libraries(bench PRIVATE vk_radix_sort volk::volk_headers VulkanMemoryAllocator)
endif()
