@echo off
:: Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting administrative privileges...
    powershell -Command "Start-Process '%0' -Verb RunAs"
    exit /B
)

:: The command you want to run
checknetisolation loopbackexempt -is -p=S-1-15-2-1327562418-1580525836-448808325-2949510007-1133312227-1793391558-2069661456
pause
