# GeoDubber v1 review: what went wrong in the Mickey 17 dub

## What was reviewed, and what was not available

Reviewed: the source zip, the four rotated logs, the translation output (1,279 subtitle lines), the SQLite
database (two projects with their speaker and voice assignments), and the Python code.

**Not available, so not checked:**
* the video and its audio: nothing could be listened to, so voice gender was **not** verified by ear;
* the reference recordings `voices/*/ref.wav|mp3` (missing from the zip);
* the per-project `projects/<name>/state.json` and the English source (`*_original.srt`): the `projects` folder was empty.

## Findings, by impact

### 1. The translator ran out of Gemini quota and fell back to a word-level web translator (main cause of poor text)
* Logs: about **3,450 HTTP 429 (quota) errors**, about **89 successful Gemini answers**, about **530 lines**
  translated by the web fallback (`Fell back to web translate`).
* Cause: one API request per line, up to six in parallel, no back-off, and the retry loop tried each model
  immediately. `gemini-3.8-flash` and `gemini-flash-latest` returned 429 for every request in the logs;
  only `gemini-3.5-flash-lite` ever answered.
* Fixed: batched JSON requests (16 lines each), quota back-off that honours the server's retry delay, models
  that are cooling down are skipped, and web-fallback lines are flagged (`mt_source = web`) with a warning.

### 2. Nothing checked the translation, so broken lines were accepted
Automatic checks on the output:
* 15 lines still contain English words, for example `slacked`, `This gate my gate`, `expendable`, `Arcadi 3.0`,
  `Sea Twink`, `T-Mobile`, `Purple Joe`.
* 2 repetition loops from Whisper, for example `ძვირფასი- ძვირფასი- ძვირფასი-…` (about 10 repeats).
* 299 lines of two words or fewer, many from fragmented speech recognition.
* 7 identical texts repeated 3 or more times.
* The log says **596 lines are longer than their time window even after the 1.3× speed-up**, so they overlap the
  next line. In the code, the translator's `max_seconds` value was passed in and then ignored.
* Fixed: every line is validated (present, Georgian, no stray English, not far too long). Repair and "shorter"
  requests are sent only for the lines that need them. Repetition runs are collapsed.

### 3. Garbled and mistranslated lines (examples from the output; judged without the English source)
* A line that reads like `ვირუსები ავსებენ ყველა ავიოლას` is a speech-recognition error that the translator
  turned into nonsense. The likely cause is `beam 1` in the quality preset, which the code comment itself calls
  the slow setting that was kept low.
* `მე ვიქნებოდი ბინძური მდიდარი` is a mistranslation of an idiom; it needs the English source to correct properly.
* `ლაბორატორ ბაყაყს` and `ტერმინუსი` are literal translations of names and terms that the glossary should handle.
* Fixed: `beam 5` on the quality preset, Whisper temperature fallback, the glossary, and the prompt rules
  (meaning first, profanity kept at its strength, no literal calques, timing).

### 4. Names were inconsistent, and the glossary never ran
* The `glossary` value was read from state but never set, and the code replaced terms in the **English** source.
* Example: the Ylfa character was written `ელფა` in the output. `glossary.txt` now holds the names
  (verify them against the credits).

### 5. Voice library silently empty (both reference voices)
* `voices/giorgi/voice.json` and `voices/nino/voice.json` use Georgian quotation marks (`„…“`) as JSON
  delimiters. That is invalid JSON, so `load_user_voices` skipped both files without a message.
* Fixed. The reference audio itself is still missing from the zip.

### 6. Voice gender: the decisions that can give the wrong voice
Ranked by how likely they are to cause the reported problem:
1. **Edge has one male and one female Georgian voice.** Any third character of the same gender shares a voice.
   Added Edge variants (`edge_character_variants`, on by default). These are **not tested by ear**.
2. **Pitch-rank fallback (no Gemini answer):** with three or more speakers, every "middle" speaker was forced to
   male. A female character in the middle would therefore get a male voice. Fixed: each middle speaker is decided
   by its own measured pitch.
3. **Gemini's gender label was trusted without a check.** Added a cross-check against the median pitch
   (male at 200 Hz or more, or female at 140 Hz or less). A mismatch produces a warning and a review flag; the
   voice is not changed automatically.
4. A hard-coded preference for the Edge voice IDs in the assignment code, which broke as soon as more voices existed.
   Removed.

In the saved Mickey project, the mapping is `speaker_1 → Giorgi (male)` and `speaker_2 → Nino (female)`. Without the
audio I could not check which character each speaker is. **Open Review on your PC and play the preview of
`speaker_2` before trusting it.** The Taylor Swift project in the same database also has two speakers, one of each gender.

### 7. Security
* `settings.json` contained a **live Gemini API key**. It has been removed from the shipped file, and the key
  is read from the `GEMINI_API_KEY` environment variable. **If this zip has been shared, revoke that key and
  create a new one.**

### 8. Defaults and tests
* The default translation model (`gemini-3.8-flash`) never answered in the logs. The default is now
  `gemini-3.5-flash-lite`; the translator still tries the others when that one is over quota.
* Test suite, before and after: the original code, run in this sandbox with a `google.genai` stub, failed
  9 tests in `test_pipeline` and `test_speakers_sync`. After the changes, 71 tests run and 69 pass.
  * The 2 failures left are in `test_speakers_sync.py` (gender confidence thresholds and a voice-assignment test
    that does not supply a confidence). They already failed in the original and are unchanged.
  * Three test expectations were out of date with the code and were corrected: the settings round-trip
    presets, the diarization-failure test (it never set a Hugging Face token, so the diarizer never ran), and
    the full-pipeline check, which depended on the order of parallel TTS threads.

## Recommendations not implemented (bigger changes, your decision)
* **Use real English subtitles when you have them.** The `imported_subtitles` setting skips speech recognition
  entirely, which removes most of the garbling in finding 3. A proper SRT for the film would be the largest quality gain.
* **Re-dub the film after this update.** The old Georgian lines are valid text, so the pipeline keeps them. Clear
  them first with `python scripts\reset_translations.py projects\<project>\state.json`.
* **Listen to two or three scenes per character** in Review before the full run, and check the variant voices.
* The glossary and the variant switch are in `settings.json` and `glossary.txt`, not yet in the GUI.

## Verification status
* Tests: 71 run, 69 pass. The real Gemini, Whisper, Edge and F5 calls, and the GUI were **not** run (no network,
  no PySide6, no models in this sandbox).
* The new translator logic is covered by fake-client tests (batching, repair, back-off, cancel during back-off,
  fallback flags, glossary and gender in the prompt, length retry).
