@echo off
setlocal
cd /d "%~dp0"

if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" -c "import os, fitz; src='figures/Pipeline.svg'; dst='figures/Pipeline.pdf'; os.path.exists(src) and (not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst)) and open(dst, 'wb').write(fitz.open(src).convert_to_pdf())" >nul 2>&1
)

echo Compiling Inria Poster using portable Tectonic...
"..\tectonic_bin\tectonic.exe" -X compile "inria-poster.tex" --synctex --keep-logs --keep-intermediates
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] LaTeX poster compilation failed.
    exit /b %ERRORLEVEL%
)
echo Done! Output saved to:
echo   - %~dp0inria-poster.pdf
