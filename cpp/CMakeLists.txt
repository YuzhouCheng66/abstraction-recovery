cmake_minimum_required(VERSION 3.15)
project(NdimGaussian LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_VTUNE_RELWITHDEBINFO "Enable VTune-friendly symbols and stack flags for RelWithDebInfo." ON)

# ==================== Build flags ====================
if(MSVC)
  add_definitions(-DEIGEN_NO_DEBUG)

  # ---- Common: treat Release + RelWithDebInfo as "production-optimized"
  # Use generator expressions so it works for multi-config generators (VS).
  # Keep ReleaseWithDebInfo as a compatibility alias for older local build dirs.
  foreach(cfg Release RelWithDebInfo ReleaseWithDebInfo)
    add_compile_options($<$<CONFIG:${cfg}>:/O2>)
    add_compile_options($<$<CONFIG:${cfg}>:/Ob2>)
    add_compile_options($<$<CONFIG:${cfg}>:/Gy>)
    add_compile_options($<$<CONFIG:${cfg}>:/Gw>)
    add_compile_options($<$<CONFIG:${cfg}>:/arch:AVX2>)
    add_compile_options($<$<CONFIG:${cfg}>:/DNDEBUG>)

    add_link_options($<$<CONFIG:${cfg}>:/OPT:REF>)
    add_link_options($<$<CONFIG:${cfg}>:/OPT:ICF>)
    add_link_options($<$<CONFIG:${cfg}>:/INCREMENTAL:NO>)
  endforeach()

  if(ENABLE_VTUNE_RELWITHDEBINFO)
    # ---- Debug symbols for VTune (RelWithDebInfo only)
    # /Zi + /DEBUG:FULL gives PDBs and source line resolution in VTune.
    # /Zo keeps optimized code more debuggable and /Oy- preserves frame pointers
    # for more reliable call stacks.
    foreach(cfg RelWithDebInfo ReleaseWithDebInfo)
      add_compile_options($<$<CONFIG:${cfg}>:/Zi>)
      add_compile_options($<$<CONFIG:${cfg}>:/Zo>)
      add_compile_options($<$<CONFIG:${cfg}>:/Oy->)
      add_link_options($<$<CONFIG:${cfg}>:/DEBUG>)
      add_link_options($<$<CONFIG:${cfg}>:/DEBUG:FULL>)
    endforeach()
  endif()

endif()
# =====================================================

# OpenMP
find_package(OpenMP QUIET)

# ---- vcpkg (manifest mode) install roots inside build tree ----
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
  src/slam/SyntheticSE2MG.cpp
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
target_link_directories(test_hierarchy_gbp PRIVATE "${VCPKG_LIB_DIR}")

add_executable(test_hierarchy_gbp_nonlinear
  src/slam/test_hierarchy_gbp_nonlinear.cpp
)
target_link_libraries(test_hierarchy_gbp_nonlinear PRIVATE ndim_gaussian slam_graph)
target_link_directories(test_hierarchy_gbp_nonlinear PRIVATE "${VCPKG_LIB_DIR}")

add_executable(test_synthetic_se2_mg_svd
  src/slam/test_synthetic_se2_mg_svd.cpp
)
target_link_libraries(test_synthetic_se2_mg_svd PRIVATE ndim_gaussian slam_graph)
target_link_directories(test_synthetic_se2_mg_svd PRIVATE "${VCPKG_LIB_DIR}")
