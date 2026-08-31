@echo off
setlocal EnableExtensions

rem ================================================================
rem Pilot benchmark launcher
rem 1) Save the current Windows power plan.
rem 2) Switch to High performance, run an idle-model-idle pilot sequence.
rem 3) Restore the original power plan before exiting.
rem
rem Start HWiNFO logging BEFORE launching this file.
rem Do not terminate this window with Ctrl+C; let the script finish so
rem the original power plan is restored.
rem ================================================================

rem ----- Edit these values for each pilot model. -----
set "MODEL_ID=1"
if not "%~1"=="" set "MODEL_ID=%~1"
set "IDLE_TRIALS=3"
set "INFERENCE_TRIALS=3"
set "WARMUP_COUNT=200"
set "DURATION_SEC=60"
set "COOLDOWN_SEC=60"
set "READY_WAIT_SEC=5"
set "RESULT_CSV=measurements\benchmark_runs\pilot_benchmark_runs.csv"

rem Leave CPU_CORE empty to use Windows scheduling. Once you have found
rem a P-core logical processor number, e.g. 0, set CPU_CORE=0 here.
set "CPU_CORE=0"

set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
set "BENCHMARK_SCRIPT=%SCRIPT_DIR%benchmark_onnx.py"
set "EXIT_CODE=1"
set "ORIGINAL_SCHEME="

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python environment not found: "%PYTHON_EXE%"
    goto :restore_and_exit
)
if not exist "%BENCHMARK_SCRIPT%" (
    echo [ERROR] Benchmark script not found: "%BENCHMARK_SCRIPT%"
    goto :restore_and_exit
)

rem The active-scheme output always places the GUID in token 4, including
rem on Korean Windows installations.
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
echo [1/3] Measuring idle baseline before model %MODEL_ID%.
"%PYTHON_EXE%" "%BENCHMARK_SCRIPT%" --mode idle --trials %IDLE_TRIALS% %COMMON_ARGUMENTS% --result-csv "%SCRIPT_DIR%%RESULT_CSV%"
if errorlevel 1 goto :benchmark_failed

echo [2/3] Measuring model %MODEL_ID%.
"%PYTHON_EXE%" "%BENCHMARK_SCRIPT%" --model-id %MODEL_ID% --trials %INFERENCE_TRIALS% --warmup-count %WARMUP_COUNT% --intra-op-threads 1 %COMMON_ARGUMENTS% --result-csv "%SCRIPT_DIR%%RESULT_CSV%"
if errorlevel 1 goto :benchmark_failed

echo [3/3] Measuring idle baseline after model %MODEL_ID%.
"%PYTHON_EXE%" "%BENCHMARK_SCRIPT%" --mode idle --trials %IDLE_TRIALS% %COMMON_ARGUMENTS% --result-csv "%SCRIPT_DIR%%RESULT_CSV%"
if errorlevel 1 goto :benchmark_failed

set "EXIT_CODE=0"
goto :restore_and_exit

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
    echo Benchmark completed successfully.
) else (
    echo Benchmark ended with an error. Check the messages above.
)
pause
endlocal & exit /b %EXIT_CODE%
