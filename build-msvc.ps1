param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

# Use the checked-in native MSVC projects. CMake is intentionally not used.
$msbuild = $null
$vsDevCmd = $null
$command = Get-Command msbuild.exe -ErrorAction SilentlyContinue
if ($command) {
    $msbuild = $command.Source
}

if (-not $msbuild) {
    $knownRoots = @(
        'D:\Software\VisualStudio\2022\Community',
        (Join-Path $env:ProgramFiles 'Microsoft Visual Studio\2022\Community'),
        (Join-Path $env:ProgramFiles 'Microsoft Visual Studio\2022\Professional'),
        (Join-Path $env:ProgramFiles 'Microsoft Visual Studio\2022\Enterprise')
    )
    foreach ($root in $knownRoots) {
        $candidate = Join-Path $root 'MSBuild\Current\Bin\amd64\MSBuild.exe'
        $candidateDevCmd = Join-Path $root 'Common7\Tools\VsDevCmd.bat'
        if ((Test-Path -LiteralPath $candidate) -and
            (Test-Path -LiteralPath $candidateDevCmd)) {
            $msbuild = $candidate
            $vsDevCmd = $candidateDevCmd
            break
        }
    }
}

if (-not $msbuild) {
    Write-Error 'Visual Studio with the MSVC C++ toolchain was not found.'
    exit 1
}

Write-Host "Building Astrax with MSVC ($Configuration)..." -ForegroundColor Cyan
$solution = (Resolve-Path -LiteralPath .\Astrax.sln).Path
if ($vsDevCmd) {
    $commandLine = "`"$vsDevCmd`" -arch=amd64 && `"$msbuild`" `"$solution`" /m /t:Build /p:Configuration=$Configuration /p:Platform=x64"
    & cmd.exe /d /s /c $commandLine
} else {
    & $msbuild $solution /m /t:Build /p:Configuration=$Configuration /p:Platform=x64
}
if ($LASTEXITCODE -ne 0) {
    Write-Error "MSVC build failed."
    exit $LASTEXITCODE
}

Write-Host "Build succeeded." -ForegroundColor Green

$testExe = ".\build\$Configuration\astrax_tests.exe"
if (Test-Path -LiteralPath $testExe) {
    Write-Host "Running tests: $testExe" -ForegroundColor Cyan
    & $testExe
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Tests failed."
        exit $LASTEXITCODE
    }
}

$trainExe = ".\build\$Configuration\astrax_train.exe"
if (Test-Path -LiteralPath $trainExe) {
    Write-Host "Running training: $trainExe" -ForegroundColor Cyan
    & $trainExe
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Training failed."
        exit $LASTEXITCODE
    }
}

$demoExe = ".\build\$Configuration\astrax_demo.exe"
if (Test-Path -LiteralPath $demoExe) {
    Write-Host "Running demo: $demoExe" -ForegroundColor Cyan
    & $demoExe
}


