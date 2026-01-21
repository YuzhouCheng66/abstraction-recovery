cmake_minimum_required(VERSION 3.15)
project(NdimGaussian LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==================== Build flags ====================
if(MSVC)
  add_definitions(-DEIGEN_NO_DEBUG)

  # Release optimizations (append, do not overwrite defaults)
  add_compile_options($<$<CONFIG:Release>:/Ob3>)
  add_compile_options($<$<CONFIG:Release>:/GL>)
  add_compile_options($<$<CONFIG:Release>:/arch:AVX2>)
  add_compile_options($<$<CONFIG:Release>:/DNDEBUG>)

  add_link_options($<$<CONFIG:Release>:/LTCG>)
  add_link_options($<$<CONFIG:Release>:/OPT:REF>)
  add_link_options($<$<CONFIG:Release>:/OPT:ICF>)
endif()
# =====================================================

# OpenMP
find_package(OpenMP QUIET)

# ---- vcpkg (manifest mode) install roots inside build tree ----
# With manifest mode, vcpkg installs into: <build>/vcpkg_installed/<triplet>/{include,lib,bin}
set(VCPKG_TRIPLET "x64-windows")
set(VCPKG_INSTALLED_DIR "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TRIPLET}")
set(VCPKG_INCLUDE_DIR   "${VCPKG_INSTALLED_DIR}/include")
set(VCPKG_LIB_DIR       "${VCPKG_INSTALLED_DIR}/lib")

# ==================== Core library ====================
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
    ${VCPKG_INCLUDE_DIR}
)

if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
  target_link_libraries(ndim_gaussian PUBLIC OpenMP::OpenMP_CXX)
endif()

# ==================== CHOLMOD (SuiteSparse via vcpkg) ====================
# We link the import libs directly by name, and provide the lib search path.
# Present libs in your build tree typically include:
#   cholmod.lib suitesparseconfig.lib amd.lib colamd.lib camd.lib ccolamd.lib
target_link_directories(ndim_gaussian PRIVATE "${VCPKG_LIB_DIR}")

target_link_libraries(ndim_gaussian PRIVATE
  cholmod
  suitesparseconfig
  amd
  colamd
  camd
  ccolamd
)
# ========================================================================

# ==================== SLAM graph generation library ====================
add_library(slam_graph
  src/slam/SlamGraph.cpp
  src/slam/SlamFactorGraph.cpp
)

target_include_directories(slam_graph
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
    ${VCPKG_INCLUDE_DIR}
)

target_link_libraries(slam_graph PRIVATE ndim_gaussian)

# ==================== Tests ====================
add_executable(test_hierarchy_gbp
  src/slam/test_hierarchy_gbp.cpp
)

target_link_libraries(test_hierarchy_gbp PRIVATE ndim_gaussian slam_graph)

# (Optional but harmless) also add lib search path at the executable level.
# This helps if you later move SuiteSparse linking from ndim_gaussian to the exe.
target_link_directories(test_hierarchy_gbp PRIVATE "${VCPKG_LIB_DIR}")

# Nonlinear SE(2) GN / GBP / V-loop test
add_executable(test_hierarchy_gbp_nonlinear
  src/slam/test_hierarchy_gbp_nonlinear.cpp
)

target_link_libraries(test_hierarchy_gbp_nonlinear PRIVATE ndim_gaussian slam_graph)
target_link_directories(test_hierarchy_gbp_nonlinear PRIVATE "${VCPKG_LIB_DIR}")
