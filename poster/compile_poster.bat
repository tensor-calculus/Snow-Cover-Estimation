@echo off
setlocal
cd /d "%~dp0"
echo Compiling Inria Poster using portable Tectonic...
"..\tectonic_bin\tectonic.exe" -X compile "inria-poster.tex" --synctex --keep-logs --keep-intermediates
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] LaTeX poster compilation failed.
    exit /b %ERRORLEVEL%
)
echo Done! Output saved to:
echo   - %~dp0inria-poster.pdf
