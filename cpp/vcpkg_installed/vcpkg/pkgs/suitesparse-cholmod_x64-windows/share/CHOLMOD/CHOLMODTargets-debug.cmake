#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::CHOLMOD" for configuration "Debug"
set_property(TARGET SuiteSparse::CHOLMOD APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::CHOLMOD PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/debug/lib/cholmod.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "SuiteSparse::SuiteSparseConfig;SuiteSparse::AMD;SuiteSparse::COLAMD;SuiteSparse::CAMD;SuiteSparse::CCOLAMD"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/bin/cholmod.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::CHOLMOD )
list(APPEND _cmake_import_check_files_for_SuiteSparse::CHOLMOD "${_IMPORT_PREFIX}/debug/lib/cholmod.lib" "${_IMPORT_PREFIX}/debug/bin/cholmod.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
