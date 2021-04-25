cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(spicy_cuda_nbody LANGUAGES CXX CUDA)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

include_directories('cspice/include')
find_library(cspice cspice 'cspice/lib')
find_library(csupport csupport 'cspice/lib')

add_executable(nbody main.cu)

