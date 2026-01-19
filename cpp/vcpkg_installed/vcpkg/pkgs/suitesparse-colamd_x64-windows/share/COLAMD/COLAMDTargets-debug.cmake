#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::COLAMD" for configuration "Debug"
set_property(TARGET SuiteSparse::COLAMD APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::COLAMD PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/debug/lib/colamd.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "SuiteSparse::SuiteSparseConfig"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/bin/colamd.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::COLAMD )
list(APPEND _cmake_import_check_files_for_SuiteSparse::COLAMD "${_IMPORT_PREFIX}/debug/lib/colamd.lib" "${_IMPORT_PREFIX}/debug/bin/colamd.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
