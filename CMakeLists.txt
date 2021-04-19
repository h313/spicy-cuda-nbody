cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(spicy_cuda_nbody LANGUAGES CXX CUDA)

include_directories('cspice/include')
find_library(cspice cspice 'cspice/lib')
find_library(csupport csupport 'cspice/lib')

add_executable(nbody main.cu)