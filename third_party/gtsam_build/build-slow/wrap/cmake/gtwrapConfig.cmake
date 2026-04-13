# This config file modifies CMAKE_MODULE_PATH so that the wrap cmake files may
# be included This file also allows the use of `find_package(gtwrap)` in CMake.


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was gtwrapConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

# Set the path to the Python package directory so we can add it to the PYTHONPATH.
# Used in the *Wrap.cmake files.
set_and_check(GTWRAP_PYTHON_PACKAGE_DIR ${PACKAGE_PREFIX_DIR}/lib/gtwrap)

# Load all the CMake scripts from the standard location
include(${PACKAGE_PREFIX_DIR}/lib/cmake/gtwrap/PybindWrap.cmake)
include(${PACKAGE_PREFIX_DIR}/lib/cmake/gtwrap/MatlabWrap.cmake)
include(${PACKAGE_PREFIX_DIR}/lib/cmake/gtwrap/GtwrapUtils.cmake)

# Set the variables for the wrapping scripts to be used in the build.
set_and_check(PYBIND_WRAP_SCRIPT "${PACKAGE_PREFIX_DIR}/bin/gtwrap/pybind_wrap.py")
set_and_check(MATLAB_WRAP_SCRIPT "${PACKAGE_PREFIX_DIR}/bin/gtwrap/matlab_wrap.py")

# Load the pybind11 code from the library installation path
add_subdirectory(${PACKAGE_PREFIX_DIR}/lib/gtwrap/pybind11 pybind11)

check_required_components(gtwrap)
