"""STEP 0 - run this first on your own PC (needs internet + the packages from install.bat).

It checks, with REAL models, whether Georgian speech synthesis works and how many distinct
voices you can get. It writes WAV files you must listen to; there is no automatic judge for
"does this sound like good Georgian".

Usage (from the GeoDubber folder, venv active):
    python scripts/verify_georgian_tts.py --ref-male male.wav --ref-male-text "..." \
                                          --ref-female female.wav --ref-female-text "..."
Reference clips: 5-12 s of clear Georgian speech you are allowed to use (e.g. CC0 Common Voice
clips, or your own recording). ref-text must be the exact transcript of that clip.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEST_TEXTS = ["გამარჯობა. როგორ ხარ დღეს?", "სად მიდიხარ ასე გვიან?", "ეს ამბავი არავის უთხრა, გთხოვ."]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tts_check")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ref-male"), ap.add_argument("--ref-male-text", default="")
    ap.add_argument("--ref-female"), ap.add_argument("--ref-female-text", default="")
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()

    from config.settings import Settings, sub_dir
    from core import georgian_tts as G
    from core.audio_io import write_wav
    from core.hardware import resolve_device
    from core.voices import add_reference_voice

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    device, note = resolve_device(a.device)
    print(f"device: {device} {note}")
    report = {}

    # --- F5 Georgian -----------------------------------------------------
    try:
        model_dir = sub_dir("models") / "georgian-f5-tts"
        if not a.skip_download and not (model_dir / G.F5_CKPT).exists():
            from huggingface_hub import snapshot_download
            print("downloading", G.F5_REPO, "...")
            snapshot_download(G.F5_REPO, local_dir=str(model_dir))
        be = G.F5GeorgianBackend(device=device, nfe_step=16, model_dir=model_dir)
        voices = []
        for g, ref, txt in (("male", a.ref_male, a.ref_male_text), ("female", a.ref_female, a.ref_female_text)):
            if ref:
                voices.append(add_reference_voice(f"check_{g}", g, ref, txt, consent=True,
                                                  note="verification", root=out / "voices"))
        if not voices:
            print("F5: no reference clips given -> cannot test voices (F5 needs a reference per voice).")
            report["f5_georgian"] = "needs --ref-male/--ref-female"
        for v in voices:
            for i, t in enumerate(TEST_TEXTS):
                t0 = time.time()
                wav, sr = be.synthesize(t, v)
                p = out / f"f5_{v.gender}_{i}.wav"
                write_wav(p, wav, sr)
                print(f"F5 {v.gender} #{i}: {len(wav)/sr:.2f}s audio in {time.time()-t0:.1f}s -> {p}")
            report[f"f5_{v.gender}"] = "generated (listen to the files)"
    except Exception as e:  # report everything, never hide
        print("F5 FAILED:", type(e).__name__, e)
        report["f5_georgian"] = f"FAILED: {e}"

    # --- MMS -------------------------------------------------------------
    try:
        be = G.MMSGeorgianBackend(device=device)
        v = be.builtin_voices()[0]
        wav, sr = be.synthesize(TEST_TEXTS[0], v)
        write_wav(out / "mms_kat.wav", wav, sr)
        print(f"MMS: {len(wav)/sr:.2f}s -> {out/'mms_kat.wav'}")
        report["mms_kat"] = "generated (listen)"
    except Exception as e:
        print("MMS FAILED:", type(e).__name__, e)
        report["mms_kat"] = f"FAILED: {e}"

    print("\nSUMMARY")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print("\nNow LISTEN to the files in", out.resolve(), "- only engines that sound intelligible should be used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
