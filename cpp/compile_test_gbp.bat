@echo off
cd /d "C:\Users\27118\Desktop\Imperial\abstraction-recovery\cpp"
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
cl /std:c++17 /I include /I external\eigen /EHsc src\NdimGaussian.cpp src\gbp\Factor.cpp src\gbp\VariableNode.cpp src\gbp\FactorGraph.cpp src\test_gbp.cpp /Fe:test_gbp.exe
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===== Compilation successful! =====
    echo Running test_gbp.exe:
    echo.
    test_gbp.exe
) else (
    echo Compilation failed!
)
pause
