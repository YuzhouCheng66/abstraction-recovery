[CmdletBinding()]
param(
    [ValidateSet("hotspots", "uarch-exploration", "memory-access")]
    [string]$AnalysisType = "hotspots",
    [string]$VtuneBinDir = "",
    [string]$Executable = "",
    [string]$ResultDir = "",
    [string[]]$TargetArgs = @(),
    [ValidateSet("sw", "hw")]
    [string]$SamplingMode = "sw",
    [switch]$AllowCollectionFailure,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Cyan
}

function Quote-Arg {
    param([string]$Value)
    if ($Value -match '\s') {
        return '"' + $Value + '"'
    }
    return $Value
}

function Invoke-External {
    param(
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [switch]$AllowFailure
    )

    $renderedArgs = ($Arguments | ForEach-Object { Quote-Arg $_ }) -join " "
    Write-Host "> $(Quote-Arg $FilePath) $renderedArgs" -ForegroundColor DarkGray

    $stdoutPath = [System.IO.Path]::GetTempFileName()
    $stderrPath = [System.IO.Path]::GetTempFileName()

    try {
        try {
            $process = Start-Process -FilePath $FilePath `
                -ArgumentList $Arguments `
                -NoNewWindow `
                -Wait `
                -PassThru `
                -RedirectStandardOutput $stdoutPath `
                -RedirectStandardError $stderrPath

            $exitCode = $process.ExitCode
            $stdoutText = ""
            $stderrText = ""

            if (Test-Path $stdoutPath) {
                $rawStdout = Get-Content $stdoutPath -Raw
                if ($null -ne $rawStdout) {
                    $stdoutText = $rawStdout.TrimEnd()
                }
            }
            if (Test-Path $stderrPath) {
                $rawStderr = Get-Content $stderrPath -Raw
                if ($null -ne $rawStderr) {
                    $stderrText = $rawStderr.TrimEnd()
                }
            }

            $parts = @()
            if ($stdoutText) {
                $parts += $stdoutText
            }
            if ($stderrText) {
                $parts += $stderrText
            }
            $text = $parts -join [Environment]::NewLine
        } catch {
            if (-not $AllowFailure) {
                throw
            }
            $exitCode = -1
            $text = "Command invocation failed: $($_.Exception.Message)"
        }

        if ($text) {
            Write-Host $text
        }
    } finally {
        Remove-Item $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
    }

    if (-not $AllowFailure -and $exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $FilePath $renderedArgs"
    }

    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = $text
    }
}

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Resolve-VtuneBinDir {
    param([string]$Hint)

    $candidates = @()
    if ($Hint) {
        $candidates += $Hint
    }

    $candidates += @(
        "D:\Intel\oneAPI\vtune\2025.8\bin64",
        "C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64",
        "C:\Program Files (x86)\Intel\oneAPI\vtune\2025.8\bin64"
    )

    foreach ($candidate in $candidates | Select-Object -Unique) {
        if ($candidate -and (Test-Path (Join-Path $candidate "vtune.exe"))) {
            return (Resolve-Path $candidate).Path
        }
    }

    $vtuneCommand = Get-Command vtune.exe -ErrorAction SilentlyContinue
    if ($vtuneCommand) {
        return Split-Path -Parent $vtuneCommand.Source
    }

    throw "Could not locate vtune.exe. Pass -VtuneBinDir explicitly."
}

$scriptDir = Split-Path -Parent $PSCommandPath
$cppRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
$repoRoot = (Resolve-Path (Join-Path $cppRoot "..")).Path

$resolvedVtuneBinDir = Resolve-VtuneBinDir -Hint $VtuneBinDir
$vtuneExe = Join-Path $resolvedVtuneBinDir "vtune.exe"
$sepregExe = Join-Path $resolvedVtuneBinDir "amplxe-sepreg.exe"

if (-not $Executable) {
    $Executable = Join-Path $cppRoot "build\test_synthetic_se2_mg_svd.exe"
}
$Executable = [System.IO.Path]::GetFullPath($Executable)

if (-not (Test-Path $Executable)) {
    throw "Executable not found: $Executable"
}

if (-not $TargetArgs -or $TargetArgs.Count -eq 0) {
    $defaultProblem = Join-Path $cppRoot "tmp\synthetic_se2_n512_seed0.problem"
    $TargetArgs = @(
        "--problem-file", $defaultProblem,
        "--num-outer", "3",
        "--inner-cycles", "3",
        "--pre-sweeps", "100",
        "--group-size", "20",
        "--r-reduced", "4",
        "--sync-threads", "1"
    )
}

if (-not $ResultDir) {
    $ResultDir = Join-Path $cppRoot ("tmp\vtune_" + $AnalysisType.Replace("-", "_") + "_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
}
$ResultDir = [System.IO.Path]::GetFullPath($ResultDir)

if (Test-Path $ResultDir) {
    if ($Force) {
        Remove-Item -Recurse -Force $ResultDir
    } else {
        throw "Result directory already exists: $ResultDir. Use -Force to overwrite."
    }
}

New-Item -ItemType Directory -Force $ResultDir | Out-Null

Write-Section "Environment"
Write-Host "Repo root:        $repoRoot"
Write-Host "CPP root:         $cppRoot"
Write-Host "VTune bin dir:    $resolvedVtuneBinDir"
Write-Host "Executable:       $Executable"
Write-Host "Analysis type:    $AnalysisType"
Write-Host "Result directory: $ResultDir"

$isAdmin = Test-IsAdministrator
Write-Host "Administrator:    $isAdmin"

$hypervisorPresent = $false
$csHypervisorPresent = $false
try {
    $computerInfo = Get-ComputerInfo
    $hypervisorPresent = [bool]$computerInfo.HyperVisorPresent
    $csHypervisorPresent = [bool]$computerInfo.CsHypervisorPresent
} catch {
    Write-Warning "Could not query Get-ComputerInfo: $($_.Exception.Message)"
}
Write-Host "HypervisorPresent:   $hypervisorPresent"
Write-Host "CsHypervisorPresent: $csHypervisorPresent"

$driverServices = @(Get-Service -Name "sepdrv*", "sepdal*", "vtss*" -ErrorAction SilentlyContinue)
if ($driverServices.Count -gt 0) {
    $serviceSummary = ($driverServices | ForEach-Object { "$($_.Name)=$($_.Status)" }) -join ", "
    Write-Host "Driver services:  $serviceSummary"
} else {
    $serviceSummary = "not found"
    Write-Host "Driver services:  not found"
}

Write-Section "VTune Driver Checks"
$sepregCheck = Invoke-External -FilePath $sepregExe -Arguments @("-c") -AllowFailure
$sepregStatus = Invoke-External -FilePath $sepregExe -Arguments @("-s") -AllowFailure

if (-not $isAdmin) {
    Write-Warning "Not running as administrator. Hardware event-based analyses such as memory-access/uarch-exploration may fail."
}
if ($hypervisorPresent -or $csHypervisorPresent) {
    Write-Warning "A hypervisor is present. VTune Memory Access is commonly unavailable in virtualized environments."
}
if ($sepregCheck.ExitCode -ne 0 -or $sepregStatus.ExitCode -ne 0) {
    Write-Warning "Sampling driver checks reported a non-zero exit code. Hotspots (user-mode sampling) can still work, but HEBS-based analyses may not."
}

Write-Section "VTune Collection"
$collectArgs = @(
    "-collect", $AnalysisType,
    "-result-dir", $ResultDir,
    "--",
    $Executable
)
if ($AnalysisType -eq "hotspots") {
    $collectArgs = @(
        "-collect", $AnalysisType,
        "-knob", "sampling-mode=$SamplingMode",
        "-result-dir", $ResultDir,
        "--",
        $Executable
    )
}
$collectArgs += $TargetArgs
$collectionResult = Invoke-External -FilePath $vtuneExe -Arguments $collectArgs -AllowFailure:$AllowCollectionFailure
$collectionSucceeded = ($collectionResult.ExitCode -eq 0)

Write-Section "Reports"
$summaryPath = Join-Path $ResultDir "summary.txt"
$detailPath = Join-Path $ResultDir "hotspots.txt"
$metaPath = Join-Path $ResultDir "precheck.txt"
$collectionPath = Join-Path $ResultDir "collection.txt"

Set-Content -Path $collectionPath -Value $collectionResult.Output

if ($collectionSucceeded) {
    $summaryReport = Invoke-External -FilePath $vtuneExe -Arguments @("-report", "summary", "-r", $ResultDir)
    $detailReport = Invoke-External -FilePath $vtuneExe -Arguments @("-report", "hotspots", "-r", $ResultDir)
    Set-Content -Path $summaryPath -Value $summaryReport.Output
    Set-Content -Path $detailPath -Value $detailReport.Output
} else {
    $failureText = "Collection failed for analysis type '$AnalysisType'. See collection.txt for full output."
    Write-Warning $failureText
    Set-Content -Path $summaryPath -Value $failureText
    Set-Content -Path $detailPath -Value $failureText
}

$precheckLines = @(
    "AnalysisType=$AnalysisType",
    "Administrator=$isAdmin",
    "HyperVisorPresent=$hypervisorPresent",
    "CsHypervisorPresent=$csHypervisorPresent",
    "driver_services=$serviceSummary",
    "CollectionSucceeded=$collectionSucceeded",
    "CollectionExitCode=$($collectionResult.ExitCode)",
    "",
    "[amplxe-sepreg -c]",
    $sepregCheck.Output,
    "",
    "[amplxe-sepreg -s]",
    $sepregStatus.Output
)

Set-Content -Path $metaPath -Value $precheckLines

Write-Section "Done"
Write-Host "Precheck: $metaPath"
Write-Host "Collection: $collectionPath"
Write-Host "Summary:    $summaryPath"
Write-Host "Detail:     $detailPath"
