cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(learn_cuda LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Set build type." FORCE)
endif()

include(CTest)

set(CMAKE_CUDA_ARCHITECTURES "80")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(01_saxpy 01_saxpy.cu)
add_executable(02_timers 02_timers.cu)
add_executable(03_mish 03_mish.cu)
add_executable(03_01_mish 03_01_mish.cu)
add_executable(03_02_mish 03_02_mish.cu)
add_executable(03_03_mish 03_03_mish.cu)
add_executable(03_04_mish 03_04_mish.cu)
add_executable(03_05_mish 03_05_mish.cu)

add_executable(mish_bench mish_bench.cu)
add_executable(peak_flops peak_flops.cu)

add_executable(peak_flops_x86 peak_flops_x86.cpp)
set_target_properties(peak_flops_x86 PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -march=skylake-avx512")
set_target_properties(peak_flops_x86 PROPERTIES CXX_STANDARD 17)
