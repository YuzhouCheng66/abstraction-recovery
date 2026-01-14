
cmake_minimum_required(VERSION 3.15)
project(NdimGaussian LANGUAGES CXX)
# OpenMP support
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Core library: NdimGaussian and GBP algorithm
add_library(ndim_gaussian 
  src/NdimGaussian.cpp
  src/gbp/Factor.cpp
  src/gbp/VariableNode.cpp
  src/gbp/FactorGraph.cpp
  src/gbp/HierarchyGBP.cpp
)

target_include_directories(ndim_gaussian
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
)

# SLAM graph generation library
add_library(slam_graph
    src/slam/SlamGraph.cpp
    src/slam/SlamFactorGraph.cpp
)

target_link_libraries(slam_graph PRIVATE ndim_gaussian)

target_include_directories(slam_graph
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
)

# Tests



add_executable(test_large_scale_slam src/slam/test_large_scale_slam.cpp)
target_link_libraries(test_large_scale_slam PRIVATE ndim_gaussian slam_graph)

# Add test_hierarchy_gbp executable
add_executable(test_hierarchy_gbp src/slam/test_hierarchy_gbp.cpp)
target_link_libraries(test_hierarchy_gbp PRIVATE ndim_gaussian slam_graph)


