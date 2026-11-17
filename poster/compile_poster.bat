@echo off
setlocal
cd /d "%~dp0"

where d2 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    if exist "figures\Pipeline.d2" (
        d2 "figures\Pipeline.d2" "figures\Pipeline.pdf" >nul 2>&1
        d2 "figures\Pipeline.d2" "figures\Pipeline.svg" >nul 2>&1
    )
)

echo Compiling Inria Poster using portable Tectonic...
"..\tectonic_bin\tectonic.exe" -X compile "inria-poster.tex" --synctex --keep-logs --keep-intermediates
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] LaTeX poster compilation failed.
    exit /b %ERRORLEVEL%
)
echo Done! Output saved to:
echo   - %~dp0inria-poster.pdf
