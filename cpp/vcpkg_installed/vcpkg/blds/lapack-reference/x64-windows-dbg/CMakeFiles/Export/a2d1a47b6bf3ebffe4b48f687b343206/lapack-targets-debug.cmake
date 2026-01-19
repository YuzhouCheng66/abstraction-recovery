#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "lapack" for configuration "Debug"
set_property(TARGET lapack APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(lapack PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/lapack${CMAKE_IMPORT_LIBRARY_SUFFIX}"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/liblapack.dll"
  )

list(APPEND _cmake_import_check_targets lapack )
list(APPEND _cmake_import_check_files_for_lapack "${_IMPORT_PREFIX}/lib/lapack${CMAKE_IMPORT_LIBRARY_SUFFIX}" "${_IMPORT_PREFIX}/bin/liblapack.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
