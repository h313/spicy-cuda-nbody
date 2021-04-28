project(nbody)
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

find_package(Boost REQUIRED)
find_package(Threads REQUIRED)
find_package(TBB REQUIRED COMPONENTS tbbmalloc tbbmalloc_proxy)
find_package(OpenMP)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${TBB_INCLUDE_DIR})
include_directories(${OpenMP_INCLUDE_DIRS})

include_directories("cspice/include")
find_library(CSPICE NAMES cspice.a PATHS "cspice/lib")
find_library(CSUPPORT NAMES csupport.a PATHS "cspice/lib")

add_library(geometry geometry.cpp)
target_link_libraries(geometry Boost::boost)

add_library(tree tree.cpp)
target_link_libraries(tree Threads::Threads geometry tbb OpenMP::OpenMP_CXX)

add_library(generator generator.cpp)
target_link_libraries(generator tree ${CSUPPORT} ${CSPICE})

#add_library(data data.cpp)
#target_link_libraries(data generator Boost)

add_executable(nbody main.cpp)
target_link_libraries(nbody generator Boost::boost)
