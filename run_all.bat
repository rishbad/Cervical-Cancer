@echo off
echo ============================================================
echo  Cervical Cancer Detection — Full Training Pipeline
echo ============================================================
echo.

echo [1/2] Starting SIPaKMeD training (cervical_cancer_train.py)...
echo ============================================================
python cervical_cancer_train.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] cervical_cancer_train.py failed with exit code %ERRORLEVEL%
    echo Stopping pipeline.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [1/2] SIPaKMeD training COMPLETE.
echo.
echo ============================================================
echo [2/2] Starting PapSmear training (cervical_cancer_papsmear.py)...
echo ============================================================
python cervical_cancer_papsmear.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] cervical_cancer_papsmear.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo  ALL DONE! Both pipelines completed successfully.
echo  SIPaKMeD results  ->  cervical_output\
echo  PapSmear results  ->  cervical_output_papsmear\
echo ============================================================
pause
