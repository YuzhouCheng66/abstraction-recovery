# Install script for directory: /home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/cmake

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow")
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

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/lib/cmake/GTSAMCMakeTools/GTSAMCMakeToolsConfig.cmake;/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/lib/cmake/GTSAMCMakeTools/Config.cmake.in;/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/lib/cmake/GTSAMCMakeTools/dllexport.h.in;/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/lib/cmake/GTSAMCMakeTools/GtsamBuildTypes.cmake;/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/lib/cmake/GTSAMCMakeTools/GtsamMakeConfigFile.cmake;/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/lib/cmake/GTSAMCMakeTools/GtsamTesting.cmake;/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/lib/cmake/GTSAMCMakeTools/GtsamPrinting.cmake;/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/lib/cmake/GTSAMCMakeTools/FindNumPy.cmake;/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/lib/cmake/GTSAMCMakeTools/README.html")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/lib/cmake/GTSAMCMakeTools" TYPE FILE FILES
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/cmake/GTSAMCMakeToolsConfig.cmake"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/cmake/Config.cmake.in"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/cmake/dllexport.h.in"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/cmake/GtsamBuildTypes.cmake"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/cmake/GtsamMakeConfigFile.cmake"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/cmake/GtsamTesting.cmake"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/cmake/GtsamPrinting.cmake"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/cmake/FindNumPy.cmake"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/cmake/README.html"
    )
endif()

