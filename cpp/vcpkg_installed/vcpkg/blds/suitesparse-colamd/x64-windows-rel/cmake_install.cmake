# Install script for directory: C:/Users/27118/Desktop/Imperial/abstraction-recovery/cpp/vcpkg_installed/vcpkg/blds/suitesparse-colamd/src/v7.8.3-ee8b79bb01.clean/COLAMD

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Users/27118/Desktop/Imperial/abstraction-recovery/cpp/vcpkg_installed/vcpkg/pkgs/suitesparse-colamd_x64-windows")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "OFF")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/27118/Desktop/Imperial/abstraction-recovery/cpp/vcpkg_installed/vcpkg/blds/suitesparse-colamd/x64-windows-rel/colamd.lib")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/Users/27118/Desktop/Imperial/abstraction-recovery/cpp/vcpkg_installed/vcpkg/blds/suitesparse-colamd/x64-windows-rel/colamd.dll")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/suitesparse" TYPE FILE FILES "C:/Users/27118/Desktop/Imperial/abstraction-recovery/cpp/vcpkg_installed/vcpkg/blds/suitesparse-colamd/src/v7.8.3-ee8b79bb01.clean/COLAMD/Include/colamd.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/COLAMD/COLAMDTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/COLAMD/COLAMDTargets.cmake"
         "C:/Users/27118/Desktop/Imperial/abstraction-recovery/cpp/vcpkg_installed/vcpkg/blds/suitesparse-colamd/x64-windows-rel/CMakeFiles/Export/f2a645ace93de340ead115089baa1f11/COLAMDTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/COLAMD/COLAMDTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/COLAMD/COLAMDTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/COLAMD" TYPE FILE FILES "C:/Users/27118/Desktop/Imperial/abstraction-recovery/cpp/vcpkg_installed/vcpkg/blds/suitesparse-colamd/x64-windows-rel/CMakeFiles/Export/f2a645ace93de340ead115089baa1f11/COLAMDTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/COLAMD" TYPE FILE FILES "C:/Users/27118/Desktop/Imperial/abstraction-recovery/cpp/vcpkg_installed/vcpkg/blds/suitesparse-colamd/x64-windows-rel/CMakeFiles/Export/f2a645ace93de340ead115089baa1f11/COLAMDTargets-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/COLAMD" TYPE FILE FILES
    "C:/Users/27118/Desktop/Imperial/abstraction-recovery/cpp/vcpkg_installed/vcpkg/blds/suitesparse-colamd/x64-windows-rel/target/COLAMDConfig.cmake"
    "C:/Users/27118/Desktop/Imperial/abstraction-recovery/cpp/vcpkg_installed/vcpkg/blds/suitesparse-colamd/x64-windows-rel/COLAMDConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
  file(WRITE "C:/Users/27118/Desktop/Imperial/abstraction-recovery/cpp/vcpkg_installed/vcpkg/blds/suitesparse-colamd/x64-windows-rel/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
