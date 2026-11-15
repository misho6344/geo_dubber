# GeoDubber

Windows desktop tool that dubs video dialogue into Georgian.

## v3.10

* **No more waiting on a used-up daily quota.** When Gemini says the quota resets in hours (the server delay is over
  30 minutes), the translator stops at once: each model is asked at most once, the remaining lines are left empty, and
  the log says so. Before, it waited 120 s per round, six rounds per batch.
* After the quota resets, run `scripts\reset_translations.py projects\<project>\state.json --only-web`, then Start Dubbing.
  Lines that were already translated by Gemini are kept.
* Short quota bursts (a few seconds) are still waited out as before.

## v3.9

* **Pause, Stop and Cancel are separate buttons.**
  * **Pause / Resume:** pauses the run and continues it later (unchanged).
  * **Stop (keep progress):** stops the run and keeps finished steps, like the old Cancel. Start Dubbing resumes it.
  * **Cancel and delete project:** stops the run if it is running, then deletes the project: its work files, the dubbed
    video, subtitles and report, and its entry in the project list. **The original video is never deleted.** A YouTube
    download is the original, so it is kept too. Both Stop and Cancel ask for confirmation first.
* Cancel and delete is available whenever a project is running or loaded, including after Preview Translation.

## v3.8

* **Retired model removed.** `gemini-2.5-flash` now returns 404 ("no longer available to new users") and was tried in
  every round, costing a request and a wait each time. It is no longer in either list.
* A daily quota on `gemini-3.8-flash` (429 with a cooldown of many hours) is not a fault: the 3.5 models keep working.

## v3.7

* **Gemini model names fixed.** `gemini-flash-latest` returned 404 (not found) for this API, so the speaker analysis
  skipped it. The lists now use the models your key serves (checked with `models.list`): translation tries
  `gemini-3.5-flash-lite`, `gemini-3.5-flash`, `gemini-2.5-flash`, `gemini-3.8-flash`; speaker analysis uses the
  same order without the last one first. A 429 means the key's quota is used up for that model; a 503 is temporary.
* Tests updated to count the models from the list (101 pass; the 2 known v1 failures remain).

## v3.6

* **Fix for the v3.5 pitch check, which moved too many lines** (257 in one log). Without a Gemini answer, the speakers
  come from pitch alone, so pitch-tracker octave errors (a woman's line read at half pitch) were treated as the
  wrong gender. Now:
  * a line at half or double its character's pitch (within 6%) is kept with its character;
  * if the check would move more than 20% of the measured lines, no voice is changed at all, and a warning says so.
  With no Gemini key the characters are still guessed by pitch: add the key in Settings and re-run, or check the
  samples in Review.

## v3.5

* **Voices stay on the right character for the whole film.** Before, each line's speaker was decided on its own, so one
  wrong line could send a man to the female voice (or the reverse), and the problem grew over time. Now:
  * every line's pitch is measured (up to 3 s per line);
  * a line whose pitch clearly contradicts its character's gender moves to a character of the matching gender;
    among same-gender characters, a line moves only when it is clearly closer to another character's pitch;
  * a character's gender changes only when most of its voiced speech (at least 75%) is clearly the other gender;
    this is always shown as a warning, and a gender you set yourself is never changed;
  * unclear lines take the gender of nearby clear lines, so a single unclear line does not flip.
  The warnings say how many lines were moved, and any line that has no matching character is reported, not forced.
* **The 5-second gender sample is one character only.** The sample never includes another character's speech.
  Before, it could join two of a character's lines across another character's line in between.
* **Not tested with real audio here.** The pitch logic is covered by tests with synthetic voices, and the full
  test suite passes (99 tests; the 2 known failures from v1 are still there). After updating, press Start Dubbing on
  the project again (the speakers are re-detected), and listen to each character's sample in Review.

## v3.4

* **Translation survives busy Gemini.** 503 (high demand), 504 (timeout) and 500 errors were not retried in v3.3:
  the batch gave up and those lines went to the web fallback. They now wait (10 s, or the server's delay), the model
  cools down, and the request is retried, like quota errors.
* **Speaker detection:** the log you sent came from an older build. Use v3.3 or later, which retries busy models.

## v3.3

* **Gemini speaker detection survives a busy model.** A 503 "high demand" or 429 answer is retried on the same model
  (after 5, 15 and 30 seconds, cancel still works), then the next model is tried. The warning now lists the reasons
  from all models, not just the first. The model order is `gemini-flash-latest`, `gemini-3.5-flash-lite`, `gemini-3.8-flash`.

## v3.2

* **Settings → Processing → Gemini API key:** paste the key once; it is saved in `settings.json` on this PC
  (plain text). Leave it empty if you prefer the `GEMINI_API_KEY` environment variable. The key is used for
  translation, timing rewrites and Gemini speaker detection.
* Project regenerated from the current source.

## v3.1 fixes from a real run log

* **Gemini key missing** (`Failed to initialise Gemini client`): the run used the web fallback for 93 lines and pitch-only
  gender detection. The app now warns clearly about this. **Set the `GEMINI_API_KEY` environment variable** (Windows:
  `setx GEMINI_API_KEY "your-key"`, then restart the app) and dub again after `scripts\reset_translations.py`.
* **Demucs failed** with "Maximum segment is: 7.8": the quality preset used 8. Now 7, and the separator clamps to 7.8.
  Without background separation the original speech stays faintly audible under the dub.
* **Timing fit** is skipped with a warning when there is no key (it needs Gemini to rewrite lines).

## v3 changes (translation context, timing fit, quality control)

| Area | What v3 does | Settings |
|---|---|---|
| Context-aware translation | Batches run **in order**. Each batch sees the last 8 English=>Georgian pairs, so names, running jokes and tone stay consistent. A **style note** for the film is added to the prompt. | `translation_sequential`, `translation_context_lines`, `translation_style_note` |
| Idioms and jokes | The model is asked for the Georgian expression with the same effect. Lines it is unsure about are tagged `idiom / joke / pun / culture / unclear` and listed as warnings. The tag is stored as `tr_note` for Review. | - |
| Timing fit (new stage "Timing fit") | After voice generation, each line's **real audio length** is compared with its window. Lines that still do not fit are rewritten shorter by Gemini (same meaning) and voiced again. Only the changed lines are regenerated. | `timing_fit_rounds` (0 = off) |
| Silence trimming | Dead air at the start and end of each generated line is removed, so pauses follow the original rhythm and the dub drifts less. | `trim_clip_silence` |
| Automatic quality control | After the timeline is built: **clipping**, **silent** clips, **too short** clips (words lost), **same-speaker overlap**, and **spill** of more than 0.3 s into another speaker's line. Results go to `state.json` under `qc` and as warnings. | `voice_qc` |

Not in v3 (honest list): sentence-level splitting of long lines with inserted pauses, and automatic listening
checks (the QC works on the audio levels and timing, not on speech content). Those need either the English
subtitles (`imported_subtitles`) or a listening pass by you in Review.

New tests: `tests/test_v3_quality.py` (quality functions, notes, memory, shortening) and a timing-fit
pipeline test. The suite runs 81 tests; 2 old speaker-threshold tests still fail as before.

## v2 changes (read this first)

Built from a review of a real run (Mickey 17 dub). What changed and why:

| Problem found in the v1 run | What v2 does |
|---|---|
| Gemini hit the free-tier quota (about 3,700 HTTP 429 errors in the logs). Only about 140 Gemini answers succeeded, and roughly 530 lines fell back to a word-level web translator. | Lines are sent in **batches** (one request per 16 lines, JSON in/out). Quota errors **wait for the server's retry delay**, skip models that are cooling down, and retry. The web fallback is **flagged** (`mt_source = web`) and shown as a warning. |
| Every answer was accepted as long as it contained a Georgian letter, so English words and missing lines got through. | Each line is **validated**: it must exist, contain Georgian, contain no stray English words (unless the glossary keeps the word), and lines far too long for their window get one "shorter" retry. |
| Names were not kept consistent (the glossary code was never filled in and applied to the English source). | `glossary.txt` (project folder) holds `Source = Georgian` pairs. They are given to the model in the prompt. |
| Lines that failed were kept in English and read by a Georgian voice. | Failed lines stay **empty** (silence) and are translated again on the next run. A warning says how many. |
| Whisper repeated words in loops ("word- word- word-"), and the quality preset used beam 1. | Repetition runs of 3+ are collapsed; Whisper gets temperature fallback; the quality preset uses beam 5. |
| Edge has one male and one female Georgian voice, so every extra character of the same gender shared a voice. | Extra Edge **variants** (slightly different pitch and speed) are added for each gender. Turn off with `edge_character_variants: false`. **Listen to them before relying on them.** |
| `voices/giorgi/voice.json` and `voices/nino/voice.json` used Georgian quotes (`„…“`) which are not valid JSON, so both reference voices were silently dropped. | Fixed. The reference audio files (`ref.wav` / `ref.mp3`) are NOT in the zip; put them back in the folders if you use `f5_georgian`. |
| Gemini's gender decision was never checked against the measured pitch. | A **pitch cross-check** warns when Gemini says male with a median pitch of 200 Hz or more, or female with 140 Hz or less. The character is also marked in the review. |
| The hard-coded Edge voice preference (`edge_ka_giorgi` / `edge_ka_eka`) broke when more voices existed. | Removed; voices of the right gender are chosen in list order. |
| A live Gemini API key was stored in `settings.json`. | **Removed from the shipped file.** Set the `GEMINI_API_KEY` environment variable instead. **If that key was shared or uploaded anywhere, revoke it and create a new one.** |

### How to re-dub a project with v2
1. Put your Gemini key in the `GEMINI_API_KEY` environment variable (or in Settings).
2. Optional: edit `glossary.txt` for the names in your film.
3. Clear the old translations, so they are redone with v2: `python scripts\reset_translations.py projects\<project>\state.json`
   (add `--only-web` to redo only the lines that came from the web fallback). A backup `state.json.bak` is written.
4. Start Dubbing. Read the warnings and check the flagged lines and genders in Review before you continue.

### Settings added in v2 (edit `settings.json`; not in the GUI yet)
* `translation_glossary_file` - path to a glossary (empty = `glossary.txt` in the app folder)
* `edge_character_variants` - `true` (default) adds pitch/speed variants for Edge voices; `false` = two voices only

### Tests
`python -m pytest tests` runs 71 tests (69 pass) with stand-ins for the AI models. Two tests in `tests/test_speakers_sync.py`
(gender confidence thresholds) were already failing in v1 and are unchanged. Three v1 test expectations were
out of date with the code and were corrected (presets: fast = tiny, quality = small; a Gemini-free diarization
fallback test; a test that no longer depends on thread order).

**Not verified in this build:** the real Gemini, Whisper, Edge and F5 calls, the audibility of the Edge variants, and
the GUI (PySide6 was not available). Those need a run on your PC.

**This build is the slim source folder (about 1 MB).** FFmpeg is not included — keep your existing `tools\ffmpeg` folder, or let `install.bat` install it with winget. Do not delete your `venv` when you update; copy these files over the old project.

Two voices only: **Giorgi** (male) and **Nino** (female). With **Detect characters & gender (Gemini)** on, Gemini listens to the audio and decides how many characters there are, their gender and which voice each needs; the project settings update to match. If Gemini is unavailable, male/female are split locally by pitch. Your giorgi/nino reference clips are used only by the `f5_georgian` engine, not by Edge.

Speed presets (all three are faster than before):

| Priority | Speech model | Whisper beam | TTS lines at once | Video encode |
|---|---|---|---|---|
| fast | tiny | 1 | 8 | ultrafast (only if re-encoding) |
| balanced | base | 1 | 6 | veryfast |
| quality | small | 5 | 4 | veryfast |

The video track is still copied (not re-encoded) unless you burn subtitles.

## 1. Install (Windows 10/11)

> **Read "Status & limitations" before relying on it.** The pipeline, audio processing, resume/cancel logic and
> export are implemented and tested. The AI-model integrations and the GUI were written against the libraries'
> documented APIs but could **not** be run in the authoring environment (no internet to Hugging Face, no GPU, no Windows,
> no PySide6). Verify them on your PC with the steps below.

## 1. Install (Windows 10/11)

1. Install **Python 3.11** (python.org, tick *Add to PATH*).
2. Double-click `install.bat` (creates `venv`, installs PyTorch — CUDA build if an NVIDIA GPU is found, otherwise CPU —
   and the requirements; tries `winget install Gyan.FFmpeg` if FFmpeg is missing).
3. **FFmpeg manually** (if needed): download a build from https://www.gyan.dev/ffmpeg/builds/ and put `ffmpeg.exe` and
   `ffprobe.exe` into `tools\ffmpeg\bin\` (or add them to PATH).
4. `start.bat` launches the app.

## 2. STEP 0 — verify Georgian speech BEFORE dubbing anything

Georgian TTS is the weakest link. Researched options:

| Engine | Georgian | Voices | Notes |
|---|---|---|---|
| `f5_georgian` (default) | yes — community fine-tune `tabluka90/georgian-f5-tts` of F5-TTS v1 | **unlimited**: one reference clip per voice | trained on Common Voice Georgian; quality unverified by me; check licences (F5-TTS base weights are non-commercial) |
| `piper_ka` | yes — Piper `ka_GE` (Natia) | 1 female | fast, CPU friendly; needs `piper` binary + `.onnx` |
| `mms_kat` | yes — `facebook/mms-tts-kat` | 1 | CC-BY-NC; random prosody |
| XTTS-v2 | **no** (not in its language list) | – | not used |

Run (examples need two short clips of clear Georgian speech you may use — e.g. CC0 Common Voice clips or your own voice —
plus their exact transcripts):

```
venv\Scripts\activate
python scripts\verify_georgian_tts.py --ref-male m.wav --ref-male-text "..." --ref-female f.wav --ref-female-text "..."
```

**Listen to the generated files** in `tts_check\`. If they are not intelligible Georgian, do not continue — change engine.

## 3. First run

1. **Settings → Models**: download Georgian F5-TTS, a Whisper model (small = fast) and NLLB-200 (translation).
   Sizes are shown where known; models are *not* bundled.
2. **Speaker model (optional but needed for multiple characters)**: create a free Hugging Face account, accept the terms at
   `huggingface.co/pyannote/speaker-diarization-3.1` (and `segmentation-3.0`), create a read token, paste it in
   Settings → Processing. Without it, everything is treated as ONE speaker (a warning is shown).
3. **Settings → Voices → Add voice**: add at least one **male and one female** reference voice per character you need
   (5–12 s clip + exact Georgian transcript). You must confirm you have the right to use each recording. Characters
   never silently share a voice: if voices run out you get a warning and can assign manually (or enable sharing).
4. Select a video → **Start Dubbing**. After translation the **Review** window lets you fix genders, voices and
   Georgian text; **Continue** generates the voices and builds the video.

Output (in `output\`): `<name>_GE_DUBBED.mp4`, `<name>_GE.srt`, `<name>_original.srt`, `<name>_translation.txt`,
`<name>_report.json`.

## 4. How it works / design notes

* Video stream is **copied** (no quality loss, original resolution/fps, fast). Georgian subtitles are a selectable soft
  track; "burn subtitles" re-encodes with H.264 (needs FFmpeg with libass).
* Long videos: audio extracted by FFmpeg streaming, ASR in 10-min overlapped chunks (each sentence kept once), per-line
  TTS cache, dub track rendered block-by-block (measured peak RAM ≈ 115 MB for a 3-hour timeline).
  Progress is checkpointed (`projects\<name>\state.json`); after a crash/cancel press Start Dubbing again and finished work is reused.
* Sync: lines keep their original start; may use up to 1.5 s of following silence; otherwise sped up ≤ 1.3×
  (pitch-preserving); still-too-long lines are kept **in full**, flagged in the report and warnings.
* Background: Demucs (`--two-stems=vocals`) per 10-min chunk; if Demucs is missing the original track is kept quietly
  under the dub (warning: some original speech stays faintly audible). Background ducks while the Georgian voice speaks.
* Gender = median F0 heuristic (≈165 Hz split) with confidence; uncertain ones are highlighted. Diarization finds *voices*,
  not character names.
* Privacy: everything local; the only network use is model/YouTube downloads you start. Delete data via Projects / Settings.
* Pause takes effect at the next safe point (between chunks/lines), not mid-FFmpeg call. Cancel kills FFmpeg immediately.

## 5. Tests

`python -m pytest tests`: probing/validation, extraction, chunk planning, mixing, export, sync/placement,
block rendering, speaker assignment, gender estimator (synthetic voiced signals), voice assignment, SQLite, settings, and
full end-to-end pipeline runs including cancel→resume, pause/resume, review edits, missing-voice errors and
diarization failure — using **stand-in model backends** (tones/prefix text) so no AI model is exercised by the tests.

## 6. Build an executable

`build.bat` → `dist\GeoDubber\GeoDubber.exe` (folder build, **untested**). If PyInstaller misses a package, edit
`geodubber.spec`. The most reliable way to distribute is the portable folder + `install.bat` / `start.bat`.

## 7. Status & limitations (honest list)

Tested here (real FFmpeg/numpy code): see section 5. **Not tested** (could not run): faster-whisper, pyannote, NLLB, F5-TTS,
Piper, MMS, Demucs, yt-dlp download, CUDA, the PySide6 GUI, `install.bat`/`build.bat`/PyInstaller, burn-in subtitles.
Known risks: package version conflicts (requirements are minimums, not a verified set); F5 `infer()` argument names may differ
by version; Georgian translation quality of NLLB and naturalness of Georgian TTS are unknown until you listen/read;
gender heuristic fails for children/atypical voices; overlapping speech is not separated; no "estimated remaining time"
for the whole job is shown (only per-stage, only when measurable); several models are non-commercial.

## Troubleshooting

Logs: `logs\geodubber.log`. "FFmpeg not found" → section 1.3. "Speaker model could not be loaded" → section 3.2.
"No Georgian voice assigned" → add voices (section 3.3). Out of GPU memory → set GPU to CPU or Fast priority.
