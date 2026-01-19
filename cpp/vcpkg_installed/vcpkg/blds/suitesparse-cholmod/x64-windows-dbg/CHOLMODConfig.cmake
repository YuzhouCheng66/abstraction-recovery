#-------------------------------------------------------------------------------
# SuiteSparse/CHOLMOD/cmake_modules/CHOLMODConfig.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# CHOLMODConfig.cmake, Copyright (c) 2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the CHOLMOD include file and compiled library.
# The following targets are defined:
#   SuiteSparse::CHOLMOD           - for the shared library (if available)
#   SuiteSparse::CHOLMOD_static    - for the static library (if available)

# For backward compatibility the following variables are set:

# CHOLMOD_INCLUDE_DIR - where to find cholmod.h
# CHOLMOD_LIBRARY     - compiled CHOLMOD library
# CHOLMOD_LIBRARIES   - libraries when using CHOLMOD
# CHOLMOD_FOUND       - true if CHOLMOD found

# Set ``CMAKE_MODULE_PATH`` to the parent folder where this module file is
# installed.

#-------------------------------------------------------------------------------


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was CHOLMODConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../pkgs/suitesparse-cholmod_x64-windows/debug" ABSOLUTE)

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

set ( CHOLMOD_DATE "June 20, 2024" )
set ( CHOLMOD_VERSION_MAJOR 5 )
set ( CHOLMOD_VERSION_MINOR 3 )
set ( CHOLMOD_VERSION_PATCH 0 )
set ( CHOLMOD_VERSION "5.3.0" )

# Check for dependent targets
include ( CMakeFindDependencyMacro )
set ( _dependencies_found ON )

# Look for NVIDIA CUDA toolkit
if ( OFF AND NOT CUDAToolkit_FOUND )
    find_dependency ( CUDAToolkit  )
    if ( NOT CUDAToolkit_FOUND )
        set ( _dependencies_found OFF )
    endif ( )
endif ( )

if ( NOT _dependencies_found )
    set ( CHOLMOD_FOUND OFF )
    return ( )
endif ( )

# Look for OpenMP
if ( OFF AND NOT OpenMP_C_FOUND )
    find_dependency ( OpenMP COMPONENTS C )
    if ( NOT OpenMP_C_FOUND )
        set ( _dependencies_found OFF )
    endif ( )
endif ( )

if ( NOT _dependencies_found )
    set ( CHOLMOD_FOUND OFF )
    return ( )
endif ( )

# Look for SuiteSparse_config, COLAMD, and AMD targets
if ( on )
    if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
        # First check in a common build tree
        find_dependency ( SuiteSparse_config 7.8
            PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
        # Then, check in the currently active CMAKE_MODULE_PATH
        if ( NOT SuiteSparse_config_FOUND )
            find_dependency ( SuiteSparse_config 7.8 )
        endif ( )
    endif ( )

    if ( NOT TARGET SuiteSparse::AMD )
        # First check in a common build tree
        find_dependency ( AMD 3.3
            PATHS ${CMAKE_SOURCE_DIR}/../AMD/build NO_DEFAULT_PATH )
        # Then, check in the currently active CMAKE_MODULE_PATH
        if ( NOT AMD_FOUND )
            find_dependency ( AMD 3.3 )
        endif ( )
    endif ( )

    if ( NOT TARGET SuiteSparse::COLAMD )
        # First check in a common build tree
        find_dependency ( COLAMD 3.3
            PATHS ${CMAKE_SOURCE_DIR}/../COLAMD/build NO_DEFAULT_PATH )
        # Then, check in the currently active CMAKE_MODULE_PATH
        if ( NOT COLAMD_FOUND )
            find_dependency ( COLAMD 3.3 )
        endif ( )
    endif ( )

else ( )
    if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
        find_dependency ( SuiteSparse_config 7.8 )
    endif ( )
    if ( NOT TARGET SuiteSparse::AMD )
        find_dependency ( AMD 3.3 )
    endif ( )
    if ( NOT TARGET SuiteSparse::COLAMD )
        find_dependency ( COLAMD 3.3 )
    endif ( )
endif ( )
if ( NOT SuiteSparse_config_FOUND OR NOT AMD_FOUND OR NOT COLAMD_FOUND )
    set ( _dependencies_found OFF )
endif ( )

if ( ON )
    # If CHOLMOD was built with CAMD and CCOLAMD, look for their targets

    if ( on )
        if ( NOT TARGET SuiteSparse::CAMD )
            # First check in a common build tree
            find_dependency ( CAMD 3.3
                PATHS ${CMAKE_SOURCE_DIR}/../CAMD/build NO_DEFAULT_PATH )
            # Then, check in the currently active CMAKE_MODULE_PATH
            if ( NOT CAMD_FOUND )
                find_dependency ( CAMD 3.3 )
            endif ( )
        endif ( )

        if ( NOT TARGET SuiteSparse::CCOLAMD )
            # First check in a common build tree
            find_dependency ( CCOLAMD 3.3
                PATHS ${CMAKE_SOURCE_DIR}/../CCOLAMD/build NO_DEFAULT_PATH )
            # Then, check in the currently active CMAKE_MODULE_PATH
            if ( NOT CCOLAMD_FOUND )
                find_dependency ( CCOLAMD 3.3 )
            endif ( )
        endif ( )
    else ( )
        if ( NOT TARGET SuiteSparse::CAMD )
            find_dependency ( CAMD 3.3 )
        endif ( )
        if ( NOT TARGET SuiteSparse::CCOLAMD )
            find_dependency ( CCOLAMD 3.3 )
        endif ( )
    endif ( )
    if ( NOT CAMD_FOUND OR NOT CCOLAMD_FOUND )
       set ( _dependencies_found OFF )
    endif ( )
endif ( )

if ( NOT _dependencies_found )
    set ( CHOLMOD_FOUND OFF )
    return ( )
endif ( )


# Import target
include ( ${CMAKE_CURRENT_LIST_DIR}/CHOLMODTargets.cmake )

if ( OFF )
    if ( TARGET SuiteSparse::CHOLMOD )
        get_property ( _cholmod_aliased TARGET SuiteSparse::CHOLMOD
            PROPERTY ALIASED_TARGET )
        if ( "${_cholmod_aliased}" STREQUAL "" )
            target_include_directories ( SuiteSparse::CHOLMOD SYSTEM AFTER INTERFACE
                "$<TARGET_PROPERTY:OpenMP::OpenMP_C,INTERFACE_INCLUDE_DIRECTORIES>" )
        else ( )
            target_include_directories ( ${_cholmod_aliased} SYSTEM AFTER INTERFACE
                "$<TARGET_PROPERTY:OpenMP::OpenMP_C,INTERFACE_INCLUDE_DIRECTORIES>" )
        endif ( )
    endif ( )
    if ( TARGET SuiteSparse::CHOLMOD_static )
        get_property ( _cholmod_aliased TARGET SuiteSparse::CHOLMOD_static
            PROPERTY ALIASED_TARGET )
        if ( "${_cholmod_aliased}" STREQUAL "" )
            target_include_directories ( SuiteSparse::CHOLMOD_static SYSTEM AFTER INTERFACE
                "$<TARGET_PROPERTY:OpenMP::OpenMP_C,INTERFACE_INCLUDE_DIRECTORIES>" )
        else ( )
            target_include_directories ( ${_cholmod_aliased} SYSTEM AFTER INTERFACE
                "$<TARGET_PROPERTY:OpenMP::OpenMP_C,INTERFACE_INCLUDE_DIRECTORIES>" )
        endif ( )
    endif ( )
endif ( )

# The following is only for backward compatibility with FindCHOLMOD.

set ( _target_shared SuiteSparse::CHOLMOD )
set ( _target_static SuiteSparse::CHOLMOD_static )
set ( _var_prefix "CHOLMOD" )

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

set ( CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARY} )

macro ( suitesparse_check_exist _var _files )
  # ignore generator expressions
  string ( GENEX_STRIP "${_files}" _files2 )

  foreach ( _file ${_files2} )
    if ( NOT EXISTS "${_file}" )
      message ( FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist!" )
    endif ( )
  endforeach ()
endmacro ( )

suitesparse_check_exist ( CHOLMOD_INCLUDE_DIR ${CHOLMOD_INCLUDE_DIR} )
suitesparse_check_exist ( CHOLMOD_LIBRARY ${CHOLMOD_LIBRARY} )

message ( STATUS "CHOLMOD version: ${CHOLMOD_VERSION}" )
message ( STATUS "CHOLMOD include: ${CHOLMOD_INCLUDE_DIR}" )
message ( STATUS "CHOLMOD library: ${CHOLMOD_LIBRARY}" )
message ( STATUS "CHOLMOD static:  ${CHOLMOD_STATIC}" )
