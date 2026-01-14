@echo off
setlocal enabledelayedexpansion

REM Setup MSVC environment for x64
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo MSVC environment not found
    exit /b 1
)
call "%VCVARS%"

REM Build directory

set BUILD_DIR=build-release

REM Clean and reconfigure
if exist "%BUILD_DIR%" (
    echo Cleaning old build...
    rmdir /s /q "%BUILD_DIR%"
)

echo Creating build directory...
mkdir "%BUILD_DIR%"
cd "%BUILD_DIR%"

echo Configuring CMake for x64 with Visual Studio...
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release .
if errorlevel 1 (
    echo CMake configuration failed
    cd ..
    exit /b 1
)

cd ..
cd "%BUILD_DIR%"

echo Building test_large_scale_slam...
cmake --build . --config Release --target test_large_scale_slam
if errorlevel 1 (
    echo Build failed
    cd ..
    exit /b 1
)

echo.
echo ===============================================
echo Running large-scale SLAM convergence test...
echo ===============================================
echo.

.\Release\test_large_scale_slam.exe

cd ..
pause
