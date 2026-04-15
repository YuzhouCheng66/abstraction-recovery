[CmdletBinding()]
param(
    [string]$BuildDir = "",
    [ValidateSet("Debug", "Release", "RelWithDebInfo")]
    [string]$Config = "RelWithDebInfo",
    [int]$Jobs = 12
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-CompilerEnvCommand {
    $candidates = @(
        "D:\VS\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw "Could not locate vcvars64.bat."
}

function Resolve-MsvcToolsDir {
    param([string]$BuildDir)

    $cachePath = Join-Path $BuildDir "CMakeCache.txt"
    if (Test-Path $cachePath) {
        $compilerLine = Select-String -Path $cachePath -Pattern '^CMAKE_CXX_COMPILER:FILEPATH=(.+cl\.exe)$' | Select-Object -First 1
        if ($compilerLine) {
            $compilerPath = $compilerLine.Matches[0].Groups[1].Value
            $candidate = Split-Path (Split-Path (Split-Path $compilerPath -Parent) -Parent) -Parent
            if (Test-Path (Join-Path $candidate "lib\\x64\\msvcprt.lib")) {
                return [System.IO.Path]::GetFullPath($candidate)
            }
        }
    }

    $toolsRoot = "D:\VS\2022\BuildTools\VC\Tools\MSVC"
    $dirs = Get-ChildItem $toolsRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName "lib\\x64\\msvcprt.lib") } |
        Sort-Object Name -Descending
    if ($dirs -and $dirs.Count -gt 0) {
        return $dirs[0].FullName
    }

    throw "Could not locate an MSVC tools directory with lib\\x64\\msvcprt.lib."
}

$scriptDir = Split-Path -Parent $PSCommandPath
$cppRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
if (-not $BuildDir) {
    $BuildDir = Join-Path $cppRoot "build"
}
$BuildDir = [System.IO.Path]::GetFullPath($BuildDir)
$vcvars64 = Resolve-CompilerEnvCommand
$msvcToolsDir = Resolve-MsvcToolsDir -BuildDir $BuildDir
$msvcToolsVersion = Split-Path $msvcToolsDir -Leaf
$msvcLibDir = Join-Path $msvcToolsDir "lib\\x64"
$msvcIncludeDir = Join-Path $msvcToolsDir "include"
$msvcBinDir = Join-Path $msvcToolsDir "bin\\Hostx64\\x64"

$batchPath = Join-Path $env:TEMP ("codex_build_cpp_" + [Guid]::NewGuid().ToString("N") + ".cmd")
$batchLines = @(
    "@echo off",
    ('call "' + $vcvars64 + '"'),
    "if errorlevel 1 exit /b %errorlevel%",
    ('set "VCToolsInstallDir=' + $msvcToolsDir + '\\"'),
    ('set "VCToolsVersion=' + $msvcToolsVersion + '"'),
    ('set "PATH=' + $msvcBinDir + ';%PATH%"'),
    ('set "INCLUDE=' + $msvcIncludeDir + ';%INCLUDE%"'),
    ('set "LIB=' + $msvcLibDir + ';%LIB%"'),
    ('cmake -S "' + $cppRoot + '" -B "' + $BuildDir + '" -G Ninja -DCMAKE_BUILD_TYPE=' + $Config),
    "if errorlevel 1 exit /b %errorlevel%",
    ('cmake --build "' + $BuildDir + '" -j ' + $Jobs),
    "exit /b %errorlevel%"
)

Set-Content -Path $batchPath -Value $batchLines -Encoding ASCII

try {
    Write-Host "> cmd.exe /d /s /c $batchPath" -ForegroundColor DarkGray
    $process = Start-Process -FilePath "cmd.exe" `
        -ArgumentList @("/d", "/s", "/c", $batchPath) `
        -NoNewWindow `
        -Wait `
        -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Build failed with exit code $($process.ExitCode)."
    }
} finally {
    Remove-Item $batchPath -Force -ErrorAction SilentlyContinue
}
