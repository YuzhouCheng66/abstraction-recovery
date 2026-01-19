@echo off
REM 自动查找并设置MSVC编译环境，然后编译并运行test_hierarchy_gbp.cpp
REM 支持VS2019和VS2022常见安装路径

setlocal
set VSWHERE="%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

REM 查找最新VS安装路径
for /f "usebackq tokens=*" %%i in (`%VSWHERE% -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSPATH=%%i

if not defined VSPATH (
    echo 未找到Visual Studio安装路径，请手动设置vcvars64.bat路径。
    pause
    exit /b 1
)

set VCVARS="%VSPATH%\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
    echo 未找到vcvars64.bat，请检查Visual Studio安装。
    pause
    exit /b 1
)

call %VCVARS%

cl /Iinclude /Iexternal/eigen /EHsc src\slam\test_hierarchy_gbp.cpp /Fe:test_hierarchy_gbp.exe
if exist test_hierarchy_gbp.exe (
    test_hierarchy_gbp.exe
) else (
    echo 编译失败！
)
pause
