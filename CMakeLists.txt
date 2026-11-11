cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61 70 75)  # Pascal, Volta, Turing
endif()
project(nvve LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig)
set(DEP ${CMAKE_SOURCE_DIR}/Dependencies)
find_library(NVCUVID nvcuvid PATHS ${DEP}/Lib/linux/stubs/x86_64 REQUIRED)

add_executable(nvve
  src/main.cpp src/log.cpp src/source.cpp src/codec.cpp src/cpu.cpp src/kernels.cu
  ${DEP}/NvCodec/NvEncoder/NvEncoder.cpp
  ${DEP}/NvCodec/NvEncoder/NvEncoderCuda.cpp
  ${DEP}/NvCodec/NvDecoder/NvDecoder.cpp)
target_include_directories(nvve PRIVATE ${DEP}/include ${DEP}/NvCodec ${DEP}/Utils)
target_link_libraries(nvve PRIVATE CUDA::cudart_static CUDA::cuda_driver ${NVCUVID} ${CMAKE_DL_LIBS} pthread)
target_compile_options(nvve PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
set_source_files_properties(src/main.cpp src/log.cpp src/source.cpp src/codec.cpp src/cpu.cpp src/gui.cpp
  PROPERTIES COMPILE_OPTIONS "-Wall;-Wno-sign-compare")

if(PkgConfig_FOUND)
  pkg_check_modules(SDL2 IMPORTED_TARGET sdl2)
  pkg_check_modules(X264 IMPORTED_TARGET x264)
endif()
if(SDL2_FOUND)
  target_sources(nvve PRIVATE src/gui.cpp)
  find_package(X11 REQUIRED)
  target_link_libraries(nvve PRIVATE PkgConfig::SDL2 X11::X11)
  target_compile_definitions(nvve PRIVATE HAVE_SDL)
else()
  message(WARNING "SDL2 not found, -gui disabled")
endif()
if(X264_FOUND)
  target_link_libraries(nvve PRIVATE PkgConfig::X264)
  target_compile_definitions(nvve PRIVATE HAVE_X264)
else()
  message(WARNING "libx264 not found, -benchmark runs the GPU side only")
endif()
