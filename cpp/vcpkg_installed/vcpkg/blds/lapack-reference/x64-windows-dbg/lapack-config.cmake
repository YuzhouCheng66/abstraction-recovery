# Load lapack targets from the build tree if necessary.
set(_LAPACK_TARGET "lapack")
if(_LAPACK_TARGET AND NOT TARGET "${_LAPACK_TARGET}")
  include("C:/Users/27118/Desktop/Imperial/abstraction-recovery/cpp/vcpkg_installed/vcpkg/blds/lapack-reference/x64-windows-dbg/lapack-targets.cmake")
endif()
unset(_LAPACK_TARGET)

# Hint for project building against lapack
set(LAPACK_Fortran_COMPILER_ID "GNU")

# Report the blas and lapack raw or imported libraries.
set(LAPACK_blas_LIBRARIES "C:/Users/27118/Desktop/Imperial/abstraction-recovery/cpp/vcpkg_installed/x64-windows/debug/lib/openblas.lib")
set(LAPACK_lapack_LIBRARIES "lapack")
set(LAPACK_LIBRARIES ${LAPACK_blas_LIBRARIES} ${LAPACK_lapack_LIBRARIES})
