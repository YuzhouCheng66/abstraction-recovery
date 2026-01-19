@echo off
set VSCMD_START_DIR=.
call "D:\VS\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
lib /machine:"amd64" %*
