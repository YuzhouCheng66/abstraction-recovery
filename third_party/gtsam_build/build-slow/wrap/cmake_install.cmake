# Install script for directory: /home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/wrap

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
   "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/lib/cmake/gtwrap/gtwrapConfig.cmake;/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/lib/cmake/gtwrap/MatlabWrap.cmake;/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/lib/cmake/gtwrap/PybindWrap.cmake;/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/lib/cmake/gtwrap/GtwrapUtils.cmake")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/lib/cmake/gtwrap" TYPE FILE FILES
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/build-slow/wrap/cmake/gtwrapConfig.cmake"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/wrap/cmake/MatlabWrap.cmake"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/wrap/cmake/PybindWrap.cmake"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/wrap/cmake/GtwrapUtils.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/lib/gtwrap/gtwrap")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/lib/gtwrap" TYPE DIRECTORY FILES "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/wrap/gtwrap")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/bin/gtwrap/pybind_wrap.py;/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/bin/gtwrap/matlab_wrap.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/bin/gtwrap" TYPE PROGRAM FILES
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/wrap/scripts/pybind_wrap.py"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/wrap/scripts/matlab_wrap.py"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/include/gtwrap/matlab.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/install-slow/include/gtwrap" TYPE FILE FILES "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/wrap/matlab.h")
endif()

