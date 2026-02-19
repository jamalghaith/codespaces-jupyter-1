@echo off
setlocal enabledelayedexpansion

rem max parallel jobs (change as needed)
set MAX=24

for %%i in (*.inp) do (
  rem prepare sample folder and files
  mkdir "%%~ni"
  copy "%%~ni.inp" "%%~ni\trvmc95.inp" >NUL

  rem wait until running TRIM-* window count is below MAX
  :wait_slot_%%~ni
  rem count processes whose window title contains TRIM-
  for /f "tokens=*" %%C in ('tasklist /v ^| findstr /I "TRIM-" ^| find /c /v ""') do set RUNNING=%%C
  if not defined RUNNING set RUNNING=0
  if %RUNNING% GEQ %MAX% (
    timeout /t 1 >NUL
    goto wait_slot_%%~ni
  )

  echo.
  echo Starting sample (parallel slot): %%~ni
  echo.

  rem start background cmd that runs the per-sample sequence; the window title starts with TRIM- so we can count it
  start "TRIM-%%~ni" /high /MIN cmd /c "cd /d \"%CD%\%%~ni\" & echo Starte TRIM-SP: %%~ni & \"%~dp0trvmc95pc.exe\" & move \"trvmc95.inp\" \"%%~ni.inp\" >NUL & move \"trvmc95.out\" \"%%~ni.out\" >NUL & \"%~dp0TrimSPAuswertung.exe\" -q fort.17 \"%%~ni\" & move \"fort.17\" \"%%~ni.fort.17\" >NUL"
)

rem optional: wait for all TRIM-* windows to finish before exiting
:wait_all
for /f "tokens=*" %%D in ('tasklist /v ^| findstr /I "TRIM-" ^| find /c /v ""') do set RUNNING=%%D
if not defined RUNNING set RUNNING=0
if %RUNNING% GTR 0 (
  timeout /t 2 >NUL
  goto wait_all
)

echo All done.
endlocal