@echo off
setlocal enableextensions
cd /d "%~dp0"
echo ============================================================
echo  GeoDubber installer
echo ============================================================

where py >nul 2>nul || (echo Python launcher 'py' not found. Install Python 3.11 from python.org ^(tick "Add to PATH"^) and run again. & pause & exit /b 1)
py -3.11 --version >nul 2>nul || (echo Python 3.11 not found. Install it from python.org and run again. & pause & exit /b 1)

if not exist venv (
  echo Creating virtual environment...
  py -3.11 -m venv venv || (echo Failed to create venv & pause & exit /b 1)
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip wheel || goto :fail

where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  echo NVIDIA GPU detected - installing CUDA build of PyTorch...
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 || goto :fail
) else (
  echo No NVIDIA GPU detected - installing CPU build of PyTorch ^(processing will be much slower^)...
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu || goto :fail
)

pip install -r requirements.txt || goto :fail
pip install pytest || goto :fail

where ffmpeg >nul 2>nul
if not %errorlevel%==0 (
  echo FFmpeg not found. Trying to install it with winget...
  winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
  echo If FFmpeg was just installed, close this window and open a new one, or copy ffmpeg.exe/ffprobe.exe into tools\ffmpeg\bin
)

echo.
echo Installation finished. Next steps:
echo   1. start.bat
echo   2. Settings ^> Models: download the models
echo   3. Settings ^> Voices: add reference voices ^(see README^)
echo   4. Optional but recommended BEFORE dubbing: python scripts\verify_georgian_tts.py --help
pause
exit /b 0
:fail
echo.
echo INSTALLATION FAILED - see the messages above. Details: README.md "Troubleshooting".
pause
exit /b 1
