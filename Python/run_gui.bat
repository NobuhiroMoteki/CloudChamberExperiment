@echo off
rem Launch the cloud chamber GUI inside the "cloudchamber" conda environment.
rem (Running python.exe without activating the environment breaks MKL/LAPACK.)
cd /d "%~dp0"
call "%USERPROFILE%\miniforge3\Scripts\activate.bat" cloudchamber
python -m cloudchamber.gui
if errorlevel 1 pause
