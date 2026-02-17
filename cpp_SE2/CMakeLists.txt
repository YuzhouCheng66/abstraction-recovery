include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
  ${CMAKE_CURRENT_SOURCE_DIR}/external/spdlog/include
  ${CMAKE_CURRENT_SOURCE_DIR}/external/manif/include
  ${CMAKE_CURRENT_SOURCE_DIR}/external/tl_optional/include
  ${VCPKG_INCLUDE_DIR}
)


add_executable(test_gbp
  src/test.cpp
)

# C++ 标准（你代码里用到 manif/Eigen，建议 17）
target_compile_features(test_gbp PRIVATE cxx_std_17)


# MSVC 常用警告/优化（可选）
if(MSVC)
  target_compile_options(test_gbp PRIVATE /W3 /permissive- /EHsc)
  # 如果你是 Release 配置，/O2 会由 CMAKE_BUILD_TYPE 或 VS 配置自动处理
endif()
