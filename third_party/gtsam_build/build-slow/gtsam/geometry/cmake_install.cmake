# Install script for directory: /home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gtsam/geometry" TYPE FILE FILES
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/BearingRange.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Cal3.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Cal3Bundler.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Cal3DS2.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Cal3DS2_Base.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Cal3Fisheye.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Cal3Unified.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Cal3_S2.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Cal3_S2Stereo.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/CalibratedCamera.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/CameraSet.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Cyclic.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/EssentialMatrix.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Line3.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/OrientedPlane3.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/PinholeCamera.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/PinholePose.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/PinholeSet.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Point2.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Point3.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Pose2.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Pose3.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Quaternion.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Rot2.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Rot3.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/SO3.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/SO4.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/SOn-inl.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/SOn.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Similarity2.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Similarity3.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/SimpleCamera.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/SphericalCamera.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/StereoCamera.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/StereoPoint2.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/Unit3.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/concepts.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/geometry/triangulation.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/build-slow/gtsam/geometry/tests/cmake_install.cmake")
endif()

