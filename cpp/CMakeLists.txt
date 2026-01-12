cmake_minimum_required(VERSION 3.15)
project(NdimGaussian LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Core library: NdimGaussian and GBP algorithm
add_library(ndim_gaussian 
    src/NdimGaussian.cpp
    src/gbp/Factor.cpp
    src/gbp/VariableNode.cpp
    src/gbp/FactorGraph.cpp
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
add_executable(test_gaussian src/test_gaussian.cpp)
target_link_libraries(test_gaussian PRIVATE ndim_gaussian)

add_executable(test_gbp src/test_gbp.cpp)
target_link_libraries(test_gbp PRIVATE ndim_gaussian)

add_executable(test_slam_graph src/slam/test_slam_graph.cpp)
target_link_libraries(test_slam_graph PRIVATE ndim_gaussian slam_graph)

add_executable(test_large_scale_slam src/slam/test_large_scale_slam.cpp)
target_link_libraries(test_large_scale_slam PRIVATE ndim_gaussian slam_graph)

add_executable(test_small_slam src/slam/test_small_slam.cpp)
target_link_libraries(test_small_slam PRIVATE ndim_gaussian slam_graph)

add_executable(test_anchor_factor src/slam/test_anchor_factor.cpp)
target_link_libraries(test_anchor_factor PRIVATE ndim_gaussian slam_graph)
