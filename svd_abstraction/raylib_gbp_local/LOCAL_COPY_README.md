This directory is a local editable mirror of `/home/yuzhou/Desktop/raylib_gbp`.

Current `svd_abstraction` raylib-facing experiments prefer this local copy by
default and only fall back to the external path if this directory is missing.

Intended workflow:
- modify files here freely
- run `svd_abstraction` benchmarks against this local copy
- keep the external `raylib_gbp` repo untouched unless explicitly needed
