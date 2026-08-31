@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ================================================================
rem Production benchmark launcher
rem Runs: idle x3 -> a batch of models -> idle x3, repeatedly.
rem The idle block between batches is shared as the after-baseline for
rem one batch and the before-baseline for the next batch.
rem ================================================================

rem ----- Edit these values for each production session. -----
set "START_ID=1"
set "END_ID=20"
if not "%~1"=="" set "START_ID=%~1"
if not "%~2"=="" set "END_ID=%~2"
set "BATCH_SIZE=10"
set "IDLE_TRIALS=3"
set "INFERENCE_TRIALS=3"
set "WARMUP_COUNT=200"
set "DURATION_SEC=60"
set "COOLDOWN_SEC=60"
set "READY_WAIT_SEC=5"
set "RESULT_CSV=measurements\benchmark_runs\production_benchmark_runs.csv"

rem P-core logical processor number. Leave empty only if affinity is not used.
set "CPU_CORE=0"

set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
set "BENCHMARK_SCRIPT=%SCRIPT_DIR%benchmark_onnx.py"
set "EXIT_CODE=1"
set "ORIGINAL_SCHEME="

if %START_ID% GTR %END_ID% (
    echo [ERROR] START_ID must not exceed END_ID.
    goto :restore_and_exit
)
if %BATCH_SIZE% LEQ 0 (
    echo [ERROR] BATCH_SIZE must be positive.
    goto :restore_and_exit
)
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python environment not found: "%PYTHON_EXE%"
    goto :restore_and_exit
)
if not exist "%BENCHMARK_SCRIPT%" (
    echo [ERROR] Benchmark script not found: "%BENCHMARK_SCRIPT%"
    goto :restore_and_exit
)

for /f "tokens=4" %%G in ('powercfg /getactivescheme') do set "ORIGINAL_SCHEME=%%G"
if not defined ORIGINAL_SCHEME (
    echo [ERROR] Could not identify the current Windows power plan.
    goto :restore_and_exit
)

echo Original power plan: %ORIGINAL_SCHEME%
echo Switching to High performance...
powercfg /setactive SCHEME_MIN
if errorlevel 1 (
    echo [ERROR] Could not activate the High performance power plan.
    goto :restore_and_exit
)

set "COMMON_ARGUMENTS=--duration-sec %DURATION_SEC% --cooldown-sec %COOLDOWN_SEC% --ready-wait-sec %READY_WAIT_SEC% --high-priority"
if defined CPU_CORE set "COMMON_ARGUMENTS=%COMMON_ARGUMENTS% --cpu-core %CPU_CORE%"

echo.
echo HWiNFO logging should already be active.
echo Measuring initial idle baseline.
call :run_idle
if errorlevel 1 goto :benchmark_failed

for /l %%M in (%START_ID%,1,%END_ID%) do (
    echo Measuring model %%M of %END_ID%.
    call :run_model %%M
    if errorlevel 1 goto :benchmark_failed

    set /a "MODELS_COMPLETED=%%M-%START_ID%+1"
    set /a "BATCH_REMAINDER=MODELS_COMPLETED %% BATCH_SIZE"
    if !BATCH_REMAINDER! EQU 0 if %%M LSS %END_ID% (
        echo Measuring shared idle baseline after model %%M.
        call :run_idle
        if errorlevel 1 goto :benchmark_failed
    )
)

echo Measuring final idle baseline.
call :run_idle
if errorlevel 1 goto :benchmark_failed

set "EXIT_CODE=0"
goto :restore_and_exit

:run_idle
"%PYTHON_EXE%" "%BENCHMARK_SCRIPT%" --mode idle --trials %IDLE_TRIALS% %COMMON_ARGUMENTS% --result-csv "%SCRIPT_DIR%%RESULT_CSV%"
exit /b %ERRORLEVEL%

:run_model
"%PYTHON_EXE%" "%BENCHMARK_SCRIPT%" --model-id %~1 --trials %INFERENCE_TRIALS% --warmup-count %WARMUP_COUNT% --intra-op-threads 1 %COMMON_ARGUMENTS% --result-csv "%SCRIPT_DIR%%RESULT_CSV%"
exit /b %ERRORLEVEL%

:benchmark_failed
set "EXIT_CODE=1"

:restore_and_exit
if defined ORIGINAL_SCHEME (
    echo.
    echo Restoring original power plan: %ORIGINAL_SCHEME%
    powercfg /setactive %ORIGINAL_SCHEME%
    if errorlevel 1 (
        echo [WARNING] Failed to restore the original power plan automatically.
        if "%EXIT_CODE%"=="0" set "EXIT_CODE=1"
    )
)

if "%EXIT_CODE%"=="0" (
    echo Production benchmark completed successfully.
) else (
    echo Production benchmark ended with an error. Check the messages above.
)
pause
endlocal & exit /b %EXIT_CODE%
