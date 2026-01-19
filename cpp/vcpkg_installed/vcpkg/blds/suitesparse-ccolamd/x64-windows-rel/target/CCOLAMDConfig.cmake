#-------------------------------------------------------------------------------
# SuiteSparse/CCOLAMD/cmake_modules/CCOLAMDConfig.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# CCOLAMDConfig.cmake, Copyright (c) 2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the CCOLAMD include file and compiled library.
# The following targets are defined:
#   SuiteSparse::CCOLAMD           - for the shared library (if available)
#   SuiteSparse::CCOLAMD_static    - for the static library (if available)

# For backward compatibility the following variables are set:

# CCOLAMD_INCLUDE_DIR - where to find ccolamd.h
# CCOLAMD_LIBRARY     - dynamic CCOLAMD library
# CCOLAMD_STATIC      - static CCOLAMD library
# CCOLAMD_LIBRARIES   - libraries when using CCOLAMD
# CCOLAMD_FOUND       - true if CCOLAMD found

# Set ``CMAKE_MODULE_PATH`` to the parent folder where this module file is
# installed.

#-------------------------------------------------------------------------------


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was CCOLAMDConfig.cmake.in                            ########

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

set ( CCOLAMD_DATE "June 20, 2024" )
set ( CCOLAMD_VERSION_MAJOR 3 )
set ( CCOLAMD_VERSION_MINOR 3 )
set ( CCOLAMD_VERSION_PATCH 4 )
set ( CCOLAMD_VERSION "3.3.4" )

# Check for dependent targets
include ( CMakeFindDependencyMacro )

# Look for SuiteSparse_config target
if ( OFF )
    if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
        # First check in a common build tree
        find_dependency ( SuiteSparse_config 7.8
            PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
        # Then, check in the currently active CMAKE_MODULE_PATH
        if ( NOT SuiteSparse_config_FOUND )
            find_dependency ( SuiteSparse_config 7.8 )
        endif ( )
    endif ( )
else ( )
    if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
        find_dependency ( SuiteSparse_config 7.8 )
    endif ( )
endif ( )
if ( NOT SuiteSparse_config_FOUND )
    set ( CCOLAMD_FOUND OFF )
    return ( )
endif ( )


# Import target
include ( ${CMAKE_CURRENT_LIST_DIR}/CCOLAMDTargets.cmake )

# The following is only for backward compatibility with FindCCOLAMD.

set ( _target_shared SuiteSparse::CCOLAMD )
set ( _target_static SuiteSparse::CCOLAMD_static )
set ( _var_prefix "CCOLAMD" )

if ( NOT ON AND NOT TARGET ${_target_shared} )
    # make sure there is always an import target without suffix )
    add_library ( ${_target_shared} ALIAS ${_target_static} )
endif ( )

get_target_property ( ${_var_prefix}_INCLUDE_DIR ${_target_shared} INTERFACE_INCLUDE_DIRECTORIES )
if ( ${_var_prefix}_INCLUDE_DIR )
    # First item in SuiteSparse targets contains the "main" header directory.
    list ( GET ${_var_prefix}_INCLUDE_DIR 0 ${_var_prefix}_INCLUDE_DIR )
endif ( )
get_target_property ( ${_var_prefix}_LIBRARY ${_target_shared} IMPORTED_IMPLIB )
if ( NOT ${_var_prefix}_LIBRARY )
    get_target_property ( _library_chk ${_target_shared} IMPORTED_LOCATION )
    if ( EXISTS ${_library_chk} )
        set ( ${_var_prefix}_LIBRARY ${_library_chk} )
    endif ( )
endif ( )
if ( TARGET ${_target_static} )
    get_target_property ( ${_var_prefix}_STATIC ${_target_static} IMPORTED_LOCATION )
endif ( )

# Check for most common build types
set ( _config_types "Debug" "Release" "RelWithDebInfo" "MinSizeRel" "None" )

get_property ( _isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG )
if ( _isMultiConfig )
    # For multi-configuration generators (e.g., Visual Studio), prefer those
    # configurations.
    list ( PREPEND _config_types ${CMAKE_CONFIGURATION_TYPES} )
else ( )
    # For single-configuration generators, prefer the current configuration.
    list ( PREPEND _config_types ${CMAKE_BUILD_TYPE} )
endif ( )

list ( REMOVE_DUPLICATES _config_types )

foreach ( _config ${_config_types} )
    string ( TOUPPER ${_config} _uc_config )
    if ( NOT ${_var_prefix}_LIBRARY )
        get_target_property ( _library_chk ${_target_shared}
            IMPORTED_IMPLIB_${_uc_config} )
        if ( EXISTS ${_library_chk} )
            set ( ${_var_prefix}_LIBRARY ${_library_chk} )
        endif ( )
    endif ( )
    if ( NOT ${_var_prefix}_LIBRARY )
        get_target_property ( _library_chk ${_target_shared}
            IMPORTED_LOCATION_${_uc_config} )
        if ( EXISTS ${_library_chk} )
            set ( ${_var_prefix}_LIBRARY ${_library_chk} )
        endif ( )
    endif ( )
    if ( TARGET ${_target_static} AND NOT ${_var_prefix}_STATIC )
        get_target_property ( _library_chk ${_target_static}
            IMPORTED_LOCATION_${_uc_config} )
        if ( EXISTS ${_library_chk} )
            set ( ${_var_prefix}_STATIC ${_library_chk} )
        endif ( )
    endif ( )
endforeach ( )

set ( CCOLAMD_LIBRARIES ${CCOLAMD_LIBRARY} )

macro ( suitesparse_check_exist _var _files )
  # ignore generator expressions
  string ( GENEX_STRIP "${_files}" _files2 )

  foreach ( _file ${_files2} )
    if ( NOT EXISTS "${_file}" )
      message ( FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist!" )
    endif ( )
  endforeach ()
endmacro ( )

suitesparse_check_exist ( CCOLAMD_INCLUDE_DIR ${CCOLAMD_INCLUDE_DIR} )
suitesparse_check_exist ( CCOLAMD_LIBRARY ${CCOLAMD_LIBRARY} )

message ( STATUS "CCOLAMD version: ${CCOLAMD_VERSION}" )
message ( STATUS "CCOLAMD include: ${CCOLAMD_INCLUDE_DIR}" )
message ( STATUS "CCOLAMD library: ${CCOLAMD_LIBRARY}" )
message ( STATUS "CCOLAMD static:  ${CCOLAMD_STATIC}" )
