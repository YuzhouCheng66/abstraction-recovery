#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::AMD" for configuration "Release"
set_property(TARGET SuiteSparse::AMD APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SuiteSparse::AMD PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/amd.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "SuiteSparse::SuiteSparseConfig"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/amd.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::AMD )
list(APPEND _cmake_import_check_files_for_SuiteSparse::AMD "${_IMPORT_PREFIX}/lib/amd.lib" "${_IMPORT_PREFIX}/bin/amd.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
