# BLA_VENDOR and BLA_STATIC are documented at:
# * https://cmake.org/cmake/help/latest/module/FindBLAS.html
# * https://cmake.org/cmake/help/latest/module/FindLAPACK.html

set(BLA_VENDOR OpenBLAS)
set(BLA_STATIC OFF)
_find_package(${ARGS})
unset(BLA_VENDOR)
unset(BLA_STATIC)
