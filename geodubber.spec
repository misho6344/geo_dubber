# PyInstaller spec - folder ("onedir") build. UNTESTED in the authoring environment (no Windows / no AI packages).
# Large AI libraries need their data files collected; adjust here if a module is reported missing at runtime.
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ["faster_whisper", "ctranslate2", "f5_tts", "transformers", "tokenizers", "yt_dlp", "pyannote.audio",
            "speechbrain", "demucs", "huggingface_hub"]:
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception as e:
        print(f"[spec] skipping {pkg}: {e}")
hiddenimports += ["PySide6.QtMultimedia"]

a = Analysis(["app.py"], pathex=["."], binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             excludes=["tkinter", "matplotlib.tests"], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="GeoDubber", console=False, disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="GeoDubber")
