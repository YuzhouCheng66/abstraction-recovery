#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::CAMD" for configuration "Debug"
set_property(TARGET SuiteSparse::CAMD APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::CAMD PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/camd.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "SuiteSparse::SuiteSparseConfig"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/camd.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::CAMD )
list(APPEND _cmake_import_check_files_for_SuiteSparse::CAMD "${_IMPORT_PREFIX}/lib/camd.lib" "${_IMPORT_PREFIX}/bin/camd.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
