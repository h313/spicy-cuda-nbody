cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
find_package(Boost REQUIRED)
find_package(Threads REQUIRED)

include_directories('cspice/include')
find_library(cspice cspice 'cspice/lib')
find_library(csupport csupport 'cspice/lib')

add_library(geometry geometry.cpp)
target_link_libraries(geometry Boost_QVM_LIBRARY)

add_library(tree tree.cpp)
target_link_libraries(tree Threads::Threads geometry)

add_executable(nbody main.cpp)
target_link_libraries(nbody)
