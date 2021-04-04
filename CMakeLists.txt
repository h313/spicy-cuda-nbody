cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

include_directories('cspice/include')
find_library(cspice cspice 'cspice/lib')
find_library(csupport csupport 'cspice/lib')


