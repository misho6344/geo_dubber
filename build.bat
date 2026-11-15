@echo off
setlocal
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (echo Run install.bat first. & pause & exit /b 1)
call venv\Scripts\activate.bat
pip install -r requirements-dev.txt || goto :fail
python -m pytest -q tests || (echo Tests failed - build aborted. & pause & exit /b 1)
pyinstaller --noconfirm --clean geodubber.spec || goto :fail
if exist tools xcopy /E /I /Y tools dist\GeoDubber\tools >nul
copy /Y README.md dist\GeoDubber\ >nul
echo.
echo Build finished: dist\GeoDubber\GeoDubber.exe  ^(portable folder - copy the whole folder^)
echo Models are NOT bundled: download them from Settings ^> Models on first run.
pause
exit /b 0
:fail
echo BUILD FAILED
pause
exit /b 1
