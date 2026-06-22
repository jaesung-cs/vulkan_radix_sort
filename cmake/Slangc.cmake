# fetch_slangc(VERSION)
# Resolves slangc and sets SLANGC_EXECUTABLE in the caller's scope.
# If VRDX_SLANGC_FROM_SDK is ON, uses the slangc bundled with the Vulkan SDK.
# Otherwise, downloads a pinned release binary from GitHub.
function(fetch_slangc VERSION)
  if(VRDX_SLANGC_FROM_SDK)
    string(REPLACE glslangValidator slangc _slangc "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}")
    if(NOT EXISTS "${_slangc}")
      message(FATAL_ERROR "slangc not found in Vulkan SDK at: ${_slangc}")
    endif()
    message(STATUS "slangc executable: ${_slangc} (Vulkan SDK)")
    set(SLANGC_EXECUTABLE "${_slangc}" PARENT_SCOPE)
    return()
  endif()

  if(WIN32)
    set(SLANG_OS "windows-x86_64")
    set(SLANG_ARCHIVE_EXT "zip")
  elseif(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
      set(SLANG_OS "macos-aarch64")
    else()
      set(SLANG_OS "macos-x86_64")
    endif()
    set(SLANG_ARCHIVE_EXT "zip")
  else()
    set(SLANG_OS "linux-x86_64")
    set(SLANG_ARCHIVE_EXT "tar.gz")
  endif()

  FetchContent_Declare(slang_bin
    URL "https://github.com/shader-slang/slang/releases/download/v${VERSION}/slang-${VERSION}-${SLANG_OS}.${SLANG_ARCHIVE_EXT}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_GetProperties(slang_bin)
  if(NOT slang_bin_POPULATED)
    FetchContent_Populate(slang_bin)
  endif()

  if(WIN32)
    set(_slangc "${slang_bin_SOURCE_DIR}/bin/slangc.exe")
  else()
    set(_slangc "${slang_bin_SOURCE_DIR}/bin/slangc")
    if(EXISTS "${_slangc}")
      file(CHMOD "${_slangc}" PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE)
    else()
      message(FATAL_ERROR
        "slangc not found at: ${_slangc}\n"
        "The archive layout for slang v${VERSION} ${SLANG_OS} may differ from bin/slangc. "
        "Inspect ${slang_bin_SOURCE_DIR} and update the path.")
    endif()
  endif()

  message(STATUS "slangc executable: ${_slangc} (v${VERSION})")
  set(SLANGC_EXECUTABLE "${_slangc}" PARENT_SCOPE)
endfunction()

# build_shader(SHADER OUTPUT [DEFINE...])
# Compiles a Slang shader to src/generated/<OUTPUT>.h containing embedded SPIR-V.
function(build_shader)
  list(POP_FRONT ARGV SHADER OUTPUT)
  list(TRANSFORM ARGV PREPEND "-D" OUTPUT_VARIABLE DEFINES)

  get_filename_component(SHADER ${SHADER} ABSOLUTE)

  add_custom_target(${OUTPUT} ALL
    COMMAND
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/slangc_to_header.py
      ${SLANGC_EXECUTABLE}
      -target spirv
      -fvk-t-shift 0 0
      -fvk-u-shift 0 0
      -o ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h
      ${DEFINES}
      ${SHADER}
    DEPENDS
      ${SHADER}
      ${CMAKE_CURRENT_SOURCE_DIR}/src/shader/constants.slang
    COMMENT "Compiling ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h"
    VERBATIM
  )
endfunction()
