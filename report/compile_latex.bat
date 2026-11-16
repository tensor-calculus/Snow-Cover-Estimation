@echo off
setlocal
cd /d "%~dp0"
echo Compiling LaTeX report using portable Tectonic...
"..\tectonic_bin\tectonic.exe" -X compile "mtp_report.tex" --synctex --keep-logs --keep-intermediates
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] LaTeX compilation failed.
    exit /b %ERRORLEVEL%
)
copy /Y "mtp_report.pdf" "254161009_MTP1_Report.pdf" >nul
echo Done! Output saved to:
echo   - %~dp0mtp_report.pdf
echo   - %~dp0254161009_MTP1_Report.pdf

