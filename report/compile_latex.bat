@echo off
echo Compiling LaTeX report using portable Tectonic...
"%~dp0..\tectonic_bin\tectonic.exe" -X compile "%~dp0mtp_report.tex" --synctex --keep-logs --keep-intermediates
"%~dp0..\tectonic_bin\tectonic.exe" -X compile "%~dp0..\archives\Phase_I_extracted\mtp_report.tex" --synctex --keep-logs --keep-intermediates
copy /Y "%~dp0mtp_report.pdf" "%~dp0254161009_MTP1_Report.pdf" >nul
echo Done! Output saved to:
echo   - %~dp0mtp_report.pdf
echo   - %~dp0..\archives\Phase_I_extracted\mtp_report.pdf
echo   - %~dp0254161009_MTP1_Report.pdf
