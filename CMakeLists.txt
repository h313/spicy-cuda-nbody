project(nbody)
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

find_package(Boost REQUIRED)
find_package(Threads REQUIRED)
find_package(TBB COMPONENTS tbbmalloc tbbmalloc_proxy)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)

include_directories('cspice/include')
find_library(cspice cspice 'cspice/lib')
find_library(csupport csupport 'cspice/lib')

add_library(geometry geometry.cpp)
target_link_libraries(geometry Boost::boost)

add_library(tree tree.cpp)
target_link_libraries(tree Threads::Threads geometry tbb)

add_executable(nbody main.cpp)
target_link_libraries(nbody tree geometry)
