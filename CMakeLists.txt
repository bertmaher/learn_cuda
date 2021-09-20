cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(learn_cuda LANGUAGES CXX CUDA)
 
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
