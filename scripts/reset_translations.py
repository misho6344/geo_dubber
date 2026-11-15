"""Clear translations of one project so that Start Dubbing translates them again with the current engine.

Usage (from the GeoDubber folder):
    python scripts\\reset_translations.py projects\\<project>\\state.json            # clear every line
    python scripts\\reset_translations.py projects\\<project>\\state.json --only-web  # clear only web-fallback lines

The original state.json is copied to state.json.bak first. ASR, speakers and voices are kept, so only the
translation (and the TTS audio that depends on it) is redone. Use it after changing glossary.txt or
the translation model.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def reset(state_path: Path, only_web: bool = False) -> tuple[int, int]:
    """Return (cleared, kept). Writes state.json in place after making state.json.bak."""
    data = json.loads(state_path.read_text(encoding="utf-8"))
    cleared = kept = 0
    for seg in data.get("segments", []):
        if not (seg.get("translation") or "").strip():
            continue
        if only_web and seg.get("mt_source") not in ("web", "failed", None):
            kept += 1
            continue
        seg["translation"] = ""
        seg.pop("mt_source", None)
        cleared += 1
    if cleared:
        shutil.copyfile(state_path, state_path.with_suffix(".json.bak"))
        # Drop cached TTS for the cleared lines: their text changed, so the cache key would not match anyway.
        data["tts"] = {k: v for k, v in data.get("tts", {}).items()
                       if k in {str(s["id"]) for s in data.get("segments", []) if (s.get("translation") or "").strip()}}
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(state_path)
    return cleared, kept


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("state", type=Path, help="projects/<name>/state.json")
    ap.add_argument("--only-web", action="store_true", help="clear only lines that came from the web fallback or failed")
    args = ap.parse_args(argv)
    if not args.state.exists():
        print(f"not found: {args.state}", file=sys.stderr)
        return 1
    cleared, kept = reset(args.state, args.only_web)
    print(f"cleared {cleared} line(s); kept {kept}. Backup: {args.state.with_suffix('.json.bak')}")
    print("Now press Start Dubbing for this project again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
