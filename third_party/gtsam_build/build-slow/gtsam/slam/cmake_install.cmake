# Install script for directory: /home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gtsam/slam" TYPE FILE FILES
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/AntiFactor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/BearingFactor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/BearingRangeFactor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/BetweenFactor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/BoundingConstraint.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/EssentialMatrixConstraint.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/EssentialMatrixFactor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/FrobeniusFactor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/GeneralSFMFactor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/InitializePose.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/InitializePose3.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/JacobianFactorQ.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/JacobianFactorQR.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/JacobianFactorSVD.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/KarcherMeanFactor-inl.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/KarcherMeanFactor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/OrientedPlane3Factor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/PoseRotationPrior.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/PoseTranslationPrior.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/PriorFactor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/ProjectionFactor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/RangeFactor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/ReferenceFrameFactor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/RegularImplicitSchurFactor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/RotateFactor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/SmartFactorBase.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/SmartFactorParams.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/SmartProjectionFactor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/SmartProjectionPoseFactor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/SmartProjectionRigFactor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/StereoFactor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/TriangulationFactor.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/dataset.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/expressions.h"
    "/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/gtsam-4.2.0-slow/gtsam/slam/lago.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build/build-slow/gtsam/slam/tests/cmake_install.cmake")
endif()

