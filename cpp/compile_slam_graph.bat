@echo off
setlocal enabledelayedexpansion

REM Setup MSVC environment
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

echo Configuring CMake...
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ..
if errorlevel 1 (
    echo CMake configuration failed
    exit /b 1
)

echo Building...
cmake --build . --config Release
if errorlevel 1 (
    echo Build failed
    exit /b 1
)

echo.
echo Build successful! Running test_slam_graph...
echo.
.\Release\test_slam_graph.exe
if errorlevel 1 (
    echo Test failed
    exit /b 1
)

echo.
echo All tests passed!
cd ..
