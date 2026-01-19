#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::CCOLAMD" for configuration "Release"
set_property(TARGET SuiteSparse::CCOLAMD APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SuiteSparse::CCOLAMD PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ccolamd.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "SuiteSparse::SuiteSparseConfig"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ccolamd.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::CCOLAMD )
list(APPEND _cmake_import_check_files_for_SuiteSparse::CCOLAMD "${_IMPORT_PREFIX}/lib/ccolamd.lib" "${_IMPORT_PREFIX}/bin/ccolamd.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
