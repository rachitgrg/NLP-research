@echo off
:: run_training.bat
:: ─────────────────────────────────────────────────────────────
:: Double-click this to start training in a standalone window.
:: This window is completely independent of Antigravity —
:: it stays alive even if the IDE crashes or restarts.
::
:: Safety features:
::   1. Disables Windows sleep/hibernate so CUDA context survives
::   2. Auto-retries on crash (CUDA reset, OOM, etc.)
::   3. Full GPU power (batch=8)
::   4. Restores original power settings when done
:: ─────────────────────────────────────────────────────────────

title YOLOv8 Training - DO NOT CLOSE THIS WINDOW
color 0A

cd /d "%~dp0"

echo ===========================================================
echo   YOLOv8 Fine-Tuning Session
echo   Started: %DATE% %TIME%
echo   DO NOT CLOSE THIS WINDOW
echo   DO NOT PUT LAPTOP TO SLEEP
echo ===========================================================
echo.

:: ── Prevent Windows from sleeping ─────────────────────────────
:: Save current power timeout values, then set them to 0 (never sleep).
:: "AC" = plugged in, "DC" = battery.
:: These are restored at the end of training.
echo [POWER] Disabling sleep and hibernate to protect CUDA...
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
powercfg /change hibernate-timeout-ac 0
powercfg /change hibernate-timeout-dc 0
powercfg /change monitor-timeout-ac 0
powercfg /change monitor-timeout-dc 0
echo [POWER] Sleep/hibernate/monitor-off disabled.
echo.

:: ── Keep the system awake using powercfg /requests override ───
:: This prevents the OS from auto-sleeping even if power plan changes.
powercfg /requestsoverride PROCESS python.exe DISPLAY SYSTEM >nul 2>&1

:: ── Training loop with auto-retry ─────────────────────────────
set RETRY_COUNT=0
set MAX_RETRIES=20

:retry
set /a RETRY_COUNT+=1

if %RETRY_COUNT% GTR %MAX_RETRIES% (
    echo.
    echo [FATAL] Exceeded %MAX_RETRIES% retries. Something is seriously wrong.
    echo         Check GPU health with: nvidia-smi
    goto cleanup
)

if %RETRY_COUNT% GTR 1 (
    echo.
    echo ──────────────────────────────────────────────
    echo   AUTO-RETRY attempt %RETRY_COUNT% of %MAX_RETRIES%
    echo   Time: %TIME%
    echo ──────────────────────────────────────────────
    echo.
)

:: Run training at full power (batch=8, the default).
:: --resume loads last.pt and continues from where it stopped.
python train.py --resume

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===========================================================
    echo   Training COMPLETED SUCCESSFULLY
    echo   Finished: %DATE% %TIME%
    echo   Best weights: %~dp0runs\train\weights\best.pt
    echo   Summary:      %~dp0results_summary.txt
    echo ===========================================================
    goto cleanup
)

:: Training crashed — wait 15 seconds for GPU to recover, then retry
echo.
echo [WARNING] Training crashed with error code %ERRORLEVEL% at %TIME%
echo           Waiting 15 seconds for GPU to recover...
timeout /t 15 /nobreak
goto retry

:: ── Restore power settings ───────────────────────────────────
:cleanup
echo.
echo [POWER] Restoring normal power settings...
powercfg /change standby-timeout-ac 30
powercfg /change standby-timeout-dc 15
powercfg /change hibernate-timeout-ac 60
powercfg /change hibernate-timeout-dc 30
powercfg /change monitor-timeout-ac 15
powercfg /change monitor-timeout-dc 10
powercfg /requestsoverride PROCESS python.exe >nul 2>&1
echo [POWER] Power settings restored.
echo.

echo Press any key to close this window...
pause >nul
