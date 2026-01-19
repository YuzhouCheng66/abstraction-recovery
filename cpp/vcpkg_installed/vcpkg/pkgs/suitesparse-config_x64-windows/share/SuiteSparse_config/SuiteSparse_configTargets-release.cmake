#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::SuiteSparseConfig" for configuration "Release"
set_property(TARGET SuiteSparse::SuiteSparseConfig APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SuiteSparse::SuiteSparseConfig PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/suitesparseconfig.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/suitesparseconfig.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::SuiteSparseConfig )
list(APPEND _cmake_import_check_files_for_SuiteSparse::SuiteSparseConfig "${_IMPORT_PREFIX}/lib/suitesparseconfig.lib" "${_IMPORT_PREFIX}/bin/suitesparseconfig.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
