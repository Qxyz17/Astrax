param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command msbuild.exe -ErrorAction SilentlyContinue)) {
    throw "msbuild.exe 未在当前环境变量中。请从 Visual Studio Developer PowerShell 运行此脚本。"
}

msbuild.exe .\Astrax.sln /m /t:Build /p:Configuration=$Configuration /p:Platform=x64
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& ".\build\$Configuration\astrax_tests.exe"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& ".\build\$Configuration\astrax_demo.exe"
