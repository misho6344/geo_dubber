# GeoDubber

**GeoDubber** is a Windows desktop application that dubs video dialogue into Georgian.

This README describes the slim source distribution. It includes the application code and installation scripts; large AI models and FFmpeg binaries are not bundled.

## Latest changes

### v3.10

- **No more waiting on an exhausted daily quota.** When Gemini reports that a quota will reset in hours (a server delay longer than 30 minutes), the translator stops trying that model immediately. Each model is requested at most once, untranslated lines are left empty, and the log records why. Previous versions could wait 120 seconds per round across six rounds per batch.
- After the quota resets, run:

  ```bash
  python scripts/reset_translations.py "projects/<project>/state.json" --only-web
  ```

  Then select **Start Dubbing**. Lines already translated by Gemini are preserved.
- Short quota interruptions of a few seconds are still retried as before.

### v3.9

- **Pause, Stop, and Cancel are separate actions.**
  - **Pause / Resume** pauses processing and continues it later.
  - **Stop (keep progress)** stops the run while preserving completed work. Select **Start Dubbing** to resume.
  - **Cancel and delete project** stops a running job and deletes the project, its work files, dubbed video, subtitles, report, and project-list entry. **The original video is never deleted.** A downloaded YouTube video is treated as the original and is kept. Stop and Cancel both request confirmation.
- Cancel and delete is available whenever a project is running or loaded, including after Preview Translation.

### v3.8

- **Retired model removed.** `gemini-2.5-flash` returned 404 (`no longer available to new users`) in the tested API environment and was removed from both model lists, avoiding a failed request and unnecessary wait on every round.
- A daily quota error on `gemini-3.8-flash` (429 with a cooldown of several hours) is not treated as a general application failure; the other configured models may still work.

### v3.7

- **Gemini model names corrected.** `gemini-flash-latest` returned 404 in the tested API environment, so speaker analysis skipped it. The configured model order was updated based on the models returned by `models.list` for the tested key:
  - Translation: `gemini-3.5-flash-lite`, `gemini-3.5-flash`, `gemini-2.5-flash`, `gemini-3.8-flash`.
  - Speaker analysis: the same configured models, with the last model ordered differently.
- A 429 response means that model's quota is exhausted for the key; a 503 response is treated as temporary.
- Tests were updated to use the model list: 101 tests passed; the two known v1 failures remained.

### v3.6

- **Pitch-check correction.** The v3.5 check could move too many lines when pitch tracking made octave errors, such as measuring a woman's line at half its actual pitch.
- A line at half or double its character's pitch (within 6%) is now kept with that character.
- If a check would move more than 20% of measured lines, no voice assignments are changed and a warning is shown.
- Without a Gemini key, character genders are still estimated from pitch. Add the key in Settings and rerun, or review the samples manually.

### v3.5

- **Voice assignments are checked across the whole film.** Previously, each line was evaluated independently, allowing one incorrect line to send a character to the wrong gendered voice. The updated checks:
  - Measure each line's pitch (up to three seconds per line).
  - Move a line when its pitch clearly contradicts its character's gender; among characters of the same gender, move it only when it is clearly closer to another character's pitch.
  - Change a character's gender only when at least 75% of its voiced speech clearly indicates the other gender. This is reported as a warning. A gender set manually is never changed.
  - Assign unclear lines using nearby clear lines, reducing the chance that one unclear line flips the assignment.
- **The five-second gender sample contains one character only.** It no longer joins two lines across another character's intervening speech.
- **Real-audio testing note:** the pitch logic was covered by synthetic-voice tests, not a full real-film listening test. The test suite had 99 tests, with the same two known v1 failures. After updating, run Start Dubbing again so speakers are detected again, then listen to each character's sample in Review.

### v3.4

- **Translation retries temporary Gemini errors.** HTTP 503 (high demand), 504 (timeout), and 500 errors were not retried in v3.3. They now trigger a wait (10 seconds or the server-provided delay), model cooldown, and retry, like quota-related errors.
- The speaker-detection log discussed for this release came from an older build; v3.3 or later includes retries for busy models.

### v3.3

- **Gemini speaker detection can recover from a busy model.** A 503 high-demand or 429 response is retried on the same model after 5, 15, and 30 seconds. Cancellation remains available during retries, then the next model is tried.
- Warnings list reasons from all attempted models. The model order in this release was `gemini-flash-latest`, `gemini-3.5-flash-lite`, and `gemini-3.8-flash`.

### v3.2

- **Settings → Processing → Gemini API key:** enter the key once; it is saved in `settings.json` on this PC as plain text.
- Leave the setting empty to use the `GEMINI_API_KEY` environment variable instead.
- The key is used for translation, timing rewrites, and Gemini speaker detection.
- The project was regenerated from the current source.

### v3.1 fixes from a real run log

- **Missing Gemini key:** when Gemini could not initialize, the run used web translation fallback for 93 lines and pitch-only gender detection. The application now warns about this. Set the `GEMINI_API_KEY` environment variable and restart the application:

  ```cmd
  setx GEMINI_API_KEY "your-key"
  ```

  After setting the key, restart the app and rerun the relevant translations with `scripts/reset_translations.py` if needed.
- **Demucs segment-length error:** the quality preset used a maximum segment length of 8, but Demucs reported a maximum of 7.8. The preset was changed to 7 and the separator clamps the value to 7.8.
- **Timing fit** is skipped with a warning when no Gemini key is configured, because Gemini is needed to rewrite lines.

## v3 — Translation context, timing fit, and quality control

| Area | What it does | Settings |
|---|---|---|
| Context-aware translation | Batches run in order. Each batch receives the previous eight English-to-Georgian pairs to help maintain names, recurring jokes, and tone. A film-specific style note is also included in the prompt. | `translation_sequential`, `translation_context_lines`, `translation_style_note` |
| Idioms and jokes | The model is asked to use a Georgian expression with the same effect. Uncertain lines are tagged as `idiom`, `joke`, `pun`, `culture`, or `unclear`; the tag is stored as `tr_note` for Review. | — |
| Timing fit | After voice generation, each line's actual audio duration is compared with its available window. Lines that do not fit may be rewritten shorter by Gemini while preserving meaning, then voiced again. Only changed lines are regenerated. | `timing_fit_rounds` (`0` disables the stage) |
| Silence trimming | Removes dead air at the start and end of generated lines so pauses better follow the original rhythm and the dub drifts less. | `trim_clip_silence` |
| Automatic quality control | Checks clipping, silent clips, clips that are too short, same-speaker overlap, and spill longer than 0.3 seconds into another speaker's line. Results are stored under `qc` in `state.json` and shown as warnings. | `voice_qc` |

**Not included in v3:** sentence-level splitting of long lines with inserted pauses, and automatic listening checks. Quality control checks audio levels and timing, not whether the spoken content is correct. These tasks require imported English subtitles (`imported_subtitles`) or a listening pass in Review.

New tests include `tests/test_v3_quality.py` for quality functions, notes, context memory, and shortening, plus a timing-fit pipeline test. At that release stage, the suite contained 81 tests; two older speaker-threshold tests were still failing as they had in v1.

## v2 — Earlier improvements

Version 2 was built after reviewing a real Mickey 17 dubbing run.

| Problem found in the v1 run | What v2 changed |
|---|---|
| Gemini free-tier quota errors caused about 3,700 HTTP 429 responses in the logs. About 140 Gemini answers succeeded, while roughly 530 lines fell back to a word-level web translator. | Lines are sent in batches (one request per 16 lines, JSON in and out). Quota errors wait for the server's retry delay, skip models that are cooling down, and retry. Web fallback is flagged with `mt_source = web` and shown as a warning. |
| Answers were accepted if they contained a Georgian letter, allowing English words and missing lines to pass. | Each line is validated: it must exist, contain Georgian, avoid stray English words unless the glossary allows them, and receive one shorter retry when it is far too long for its timing window. |
| Names were inconsistent because the glossary was not fully applied to the English source. | `glossary.txt` contains `Source = Georgian` pairs and is supplied to the model in the prompt. |
| Failed translations remained in English and were read by a Georgian voice. | Failed lines are left empty (silence) and can be translated again on the next run. A warning reports how many lines failed. |
| Whisper repeated words in loops, and the quality preset used beam size 1. | Repetition runs of three or more are collapsed; Whisper uses temperature fallback; the quality preset uses beam size 5. |
| Edge had one male and one female Georgian voice, so characters of the same gender shared a voice. | Optional Edge voice variants use small pitch and speed changes. Disable them with `edge_character_variants: false`. Listen to variants before relying on them. |
| `voices/giorgi/voice.json` and `voices/nino/voice.json` used Georgian quotation marks that are invalid in JSON, so the reference voices were silently dropped. | The JSON was corrected. Reference audio (`ref.wav` or `ref.mp3`) is not included in the ZIP; restore it to the appropriate folders when using `f5_georgian`. |
| Gemini gender decisions were not checked against measured pitch. | A pitch cross-check warns when Gemini labels a voice male with a median pitch of 200 Hz or higher, or female with a median pitch of 140 Hz or lower. The character is also marked in Review. |
| Hard-coded Edge voice preferences broke when more voices were added. | Hard-coded preferences such as `edge_ka_giorgi` and `edge_ka_eka` were removed; voices of the correct gender are selected in list order. |
| A live Gemini API key had been stored in `settings.json`. | The key was removed from the shipped file. Set `GEMINI_API_KEY` instead. If a key was shared or uploaded, revoke it and create a new one. |

### Re-dub a project with v2 or later translation logic

1. Configure `GEMINI_API_KEY` as an environment variable or enter it in Settings.
2. Optionally edit `glossary.txt` to add the names used in the film.
3. Clear old translations so they can be regenerated:

   ```bash
   python scripts/reset_translations.py "projects/<project>/state.json"
   ```

   Add `--only-web` to redo only lines that came from the web fallback. A backup named `state.json.bak` is written.
4. Select **Start Dubbing**. Review warnings and check flagged lines and character genders in Review before continuing.

### v2 settings

These settings can be edited in `settings.json` (they were not available in the GUI at the time of v2):

- `translation_glossary_file` — glossary path; an empty value uses `glossary.txt` in the application folder.
- `edge_character_variants` — `true` (default) adds pitch/speed variants for Edge voices; `false` keeps only the two standard voices.

### v2 test status

At the v2 release stage, `python -m pytest tests` ran 71 tests: 69 passed. Two tests in `tests/test_speakers_sync.py` (gender confidence thresholds) had already been failing in v1 and were unchanged. Three v1 expectations that no longer matched the code were corrected: presets (fast = tiny, quality = small), Gemini-free diarization fallback, and a test that no longer depends on thread order.

## Voices and speed presets

The bundled voice configuration is intended to use two Georgian voices:

- **Giorgi** — male.
- **Nino** — female.

With **Detect characters & gender (Gemini)** enabled, Gemini analyzes the audio and estimates the number of characters, their genders, and the voice assignment for each. Project settings are updated to match. If Gemini is unavailable, the application falls back to local male/female pitch estimation.

The Giorgi and Nino reference clips are used only by the `f5_georgian` engine, not by Edge. Additional Edge variants may be generated from the standard voices; listen to them before relying on them.

| Priority | Speech model | Whisper beam | TTS lines at once | Video encoding |
|---|---:|---:|---:|---|
| fast | tiny | 1 | 8 | ultrafast (only when re-encoding) |
| balanced | base | 1 | 6 | veryfast |
| quality | small | 5 | 4 | veryfast |

The video track is copied rather than re-encoded unless subtitles are burned into the video.

## 1. Install (Windows 10/11)

> **Read [Status and limitations](#7-status-and-limitations) before relying on the application.** The pipeline, audio processing, resume/cancel logic, and export were tested with stand-in model backends. The real AI-model integrations and GUI still need verification on your Windows PC.

1. Install **Python 3.11** from [python.org](https://www.python.org/downloads/). During setup, select **Add Python to PATH**.
2. Double-click `install.bat`. It creates a virtual environment, installs PyTorch (CUDA build if an NVIDIA GPU is detected, otherwise CPU), installs the requirements, and attempts to install FFmpeg with winget if FFmpeg is missing.
3. If FFmpeg is still missing, download a Windows build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and put `ffmpeg.exe` and `ffprobe.exe` in `tools\ffmpeg\bin\`, or add them to your Windows `PATH`.
4. Run `start.bat` to launch the application.

## 2. Step 0 — verify Georgian speech before dubbing

Georgian text-to-speech is a critical part of the result. Test the selected engine before dubbing a full film.

| Engine | Georgian support | Voices | Notes |
|---|---|---|---|
| `f5_georgian` (default) | Yes — community fine-tune `tabluka90/georgian-f5-tts` of F5-TTS v1 | Multiple, using one reference clip per voice | Trained on Common Voice Georgian. Quality has not been independently verified here. Check the model and base-model licences; F5-TTS base weights are non-commercial. |
| `piper_ka` | Yes — Piper `ka_GE` (Natia) | One female voice | Fast and CPU-friendly; requires the Piper binary and `.onnx` model. |
| `mms_kat` | Yes — `facebook/mms-tts-kat` | One voice | CC-BY-NC; prosody may sound unnatural. |
| XTTS-v2 | No | — | Georgian is not in its supported language list and this project does not use it. |

To test F5-TTS, prepare two short, clear Georgian speech clips that you have permission to use (for example, your own recordings or suitable CC0 Common Voice clips), together with exact transcripts:

```bat
venv\Scripts\activate
python scripts\verify_georgian_tts.py --ref-male m.wav --ref-male-text "..." --ref-female f.wav --ref-female-text "..."
```

Listen to the generated files in `tts_check\`. If the Georgian speech is not intelligible, do not continue with that engine; select another engine and test it.

## 3. First run

1. Open **Settings → Models** and download the Georgian F5-TTS model, a Whisper model (the `small` model is a practical starting point), and NLLB-200 for translation. Models are not bundled in the slim source distribution; sizes are shown where known.
2. **Optional speaker model:** for multi-character diarization, create a free Hugging Face account, accept the terms for [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) and its required [`segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) dependency, create a read token, and enter it in Settings → Processing. Without this model, the application treats the audio as one speaker and shows a warning.
3. Open **Settings → Voices → Add voice** and add at least one male and one female reference voice for the characters you need (a 5–12 second clip plus its exact Georgian transcript). Confirm that you have the right to use every recording. If voices run out, the application warns you; assign voices manually or enable sharing if appropriate.
4. Select a video and choose **Start Dubbing**. After translation, use **Review** to correct genders, voice assignments, and Georgian text. Select **Continue** to generate voices and build the video.

The output folder is `output\`. Expected files include:

- `<name>_GE_DUBBED.mp4`
- `<name>_GE.srt`
- `<name>_original.srt`
- `<name>_translation.txt`
- `<name>_report.json`

## 4. How it works and design notes

- **Video:** the video stream is copied without re-encoding, preserving the original resolution and frame rate. Georgian subtitles are added as a selectable soft subtitle track. Choosing **Burn subtitles** re-encodes with H.264 and requires FFmpeg with `libass`.
- **Long videos:** audio is extracted through FFmpeg streaming. ASR runs in overlapping 10-minute chunks, with each sentence retained once. TTS output is cached per line, and the dub track is rendered in blocks. Measured peak RAM was approximately 115 MB for a three-hour timeline in the tested pipeline.
- **Resume:** progress is checkpointed in `projects\<name>\state.json`. After a crash or stop, select **Start Dubbing** again; completed work is reused.
- **Synchronization:** lines keep their original start time and may use up to 1.5 seconds of following silence. Otherwise, audio can be sped up by up to 1.3× with pitch preservation. Lines that still do not fit are kept in full and flagged in the report and warnings.
- **Background audio:** Demucs can separate vocals with `--two-stems=vocals` in 10-minute chunks. If Demucs is unavailable, the original track is kept quietly under the dub, so some original speech may remain faintly audible. Background audio is ducked while Georgian speech plays.
- **Gender:** a median fundamental-frequency (F0) heuristic uses an approximate 165 Hz split and confidence checks. Uncertain assignments are highlighted. Diarization identifies voices, not character names.
- **Privacy:** processing is local by default. Network access is used for model downloads and YouTube downloads that you initiate, as well as configured online translation or AI services such as Gemini. Delete project data through Projects / Settings.
- **Pause and cancel:** Pause takes effect at the next safe point between chunks or lines, not in the middle of an FFmpeg operation. Cancel terminates FFmpeg immediately.

## 5. Tests

Run the test suite with:

```bash
python -m pytest tests
```

The tests cover probing and validation, extraction, chunk planning, mixing, export, synchronization and placement, block rendering, speaker assignment, synthetic-signal gender estimation, voice assignment, SQLite, settings, and end-to-end pipeline scenarios. These include cancel/resume, pause/resume, Review edits, missing-voice errors, and diarization failure.

The tests use stand-in model backends (tones and prefix text) so that they do not exercise real AI models.

## 6. Build an executable

Run:

```bat
build.bat
```

The expected folder build is `dist\GeoDubber\GeoDubber.exe`. This build path has not been verified on every target Windows configuration. If PyInstaller misses a package, update `geodubber.spec`.

For distribution, the portable folder together with `install.bat` and `start.bat` is the recommended approach.

## 7. Status and limitations

### Tested in the development environment

The real FFmpeg and NumPy portions of the pipeline were tested. The automated test suite covers the application logic using stand-in AI-model backends.

### Not verified in this build

The following integrations could not be run in the authoring environment and must be verified on the target Windows PC:

- faster-whisper
- pyannote speaker diarization
- NLLB translation
- F5-TTS
- Piper and MMS TTS
- Demucs
- yt-dlp downloads
- CUDA acceleration
- the PySide6 GUI
- `install.bat`, `build.bat`, and PyInstaller packaging
- subtitle burn-in
- real Gemini, Whisper, Edge, and F5 service calls

Known risks include package-version conflicts (requirements are minimum versions, not a fully verified lock), F5 `infer()` argument differences between library versions, unknown Georgian translation quality from NLLB until reviewed, unknown naturalness of generated Georgian speech until listened to, gender-estimation errors for children or atypical voices, and overlapping speech that is not separated. The application does not show a whole-job estimated remaining time; it shows stage progress and estimates only where measurable. Some model licences restrict commercial use.

## Troubleshooting

- **FFmpeg not found:** follow step 1.3 and ensure `ffmpeg.exe` and `ffprobe.exe` are on `PATH` or in `tools\ffmpeg\bin\`.
- **Speaker model could not be loaded:** follow step 3.2, accept the model terms, and verify the Hugging Face read token.
- **No Georgian voice assigned:** add voices in Settings → Voices, as described in step 3.3.
- **Out of GPU memory:** switch the processing device to CPU or select the Fast priority preset.
- **Gemini key missing:** configure `GEMINI_API_KEY` or enter the key in Settings → Processing. Do not commit API keys to Git or include them in a public or shared archive.
- **Translation quota exhausted:** wait until the quota resets, then use the v3.10 `--only-web` reset command above to retry web-fallback lines while preserving Gemini-translated lines.

Application logs are written to `logs\geodubber.log`.

---

**Distribution note:** This is the slim source folder (approximately 1 MB). FFmpeg is not included. Keep an existing `tools\ffmpeg` folder or allow `install.bat` to install FFmpeg with winget. Do not delete your existing virtual environment when updating an installation; copy the source files over the existing project only after backing up project data.
