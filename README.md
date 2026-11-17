# GeoDubber

**GeoDubber turns video dialogue into Georgian speech.** It provides a Windows desktop application and a browser-based online interface backed by a FastAPI processing service.

The project reuses one checkpointed processing pipeline. The desktop and online profiles have different runtime capabilities, so this document describes them separately.

## Contents

- [What GeoDubber does](#what-geodubber-does)
- [Desktop and online profiles](#desktop-and-online-profiles)
- [Online workflow](#online-workflow)
- [Input files and subtitles](#input-files-and-subtitles)
- [Windows desktop installation](#windows-desktop-installation)
- [Run the online API on Windows](#run-the-online-api-on-windows)
- [Connect the Admin Panel](#connect-the-admin-panel)
- [Deploy on Oracle Cloud](#deploy-on-oracle-cloud)
- [Configuration and secrets](#configuration-and-secrets)
- [Storage and backups](#storage-and-backups)
- [Troubleshooting](#troubleshooting)
- [Tests and project layout](#tests-and-project-layout)
- [Known limitations](#known-limitations)

## What GeoDubber does

GeoDubber is designed to create a Georgian-dubbed version of a video while retaining the original video track.

Depending on the selected profile and configuration, the processing pipeline can:

1. Extract audio from a video with FFmpeg.
2. Recognize dialogue with Faster-Whisper.
3. Detect speaker turns and estimate perceived voice gender.
4. Translate dialogue into Georgian, with contextual information for names, recurring expressions, jokes, and tone.
5. Let the user review translated lines, speaker assignments, and voices before synthesis.
6. Generate Georgian speech.
7. Separate dialogue from music and effects when a compatible separator is available.
8. Fit generated speech to the original timing, mix the audio, and export an MP4.

The pipeline saves project state and completed work so a stopped or interrupted job can often be resumed without repeating every completed stage.

## Desktop and online profiles

| Capability | Windows desktop | Browser / online API |
|---|---|---|
| User interface | PySide6 desktop application | Browser interface served by FastAPI |
| Processing device | CPU or supported NVIDIA CUDA setup | CPU profile |
| Speech recognition | Faster-Whisper; model selected in settings | Faster-Whisper CPU profile |
| Translation | Local/configured translation backends and Gemini when configured | Gemini-based translation when configured |
| Georgian speech | Configurable desktop engines, including F5 Georgian, Piper and MMS options | Edge Georgian TTS |
| Speaker analysis | Local diarization/pitch tools and Gemini when configured | Gemini analysis when configured, with local fallback |
| Background separation | Demucs when installed and configured | Demucs when available; otherwise a quieter original-track fallback |
| Timed script import | Supported by the processing pipeline | SRT, VTT or TXT |
| Video URL import | Depends on the desktop workflow | YouTube HTTPS, direct public HTTPS video URLs and BitTorrent Magnet links |
| Subtitle output | Configurable subtitle files and optional video subtitle track | No subtitles are embedded in the final MP4 and no SRT sidecar files are exported |
| Concurrent jobs | Desktop workflow | One online job at a time |

The online API deliberately uses a smaller runtime profile. It does not expose the desktop model downloader, desktop-only TTS engines, custom reference-voice recording, or every desktop setting.

## Online workflow

The browser interface is located in `web/index.html` and is served by `online_api.py` from the same origin. Open the API root URL in a browser to use it.

The online workflow is:

1. Enter the API token when prompted.
2. Create a project and either upload a video or provide a supported source URL.
3. Optionally upload a timed SRT, VTT or TXT script before uploading the video.
4. Choose processing options, including speed/quality priority, translation batch size, speaker count, voice gender preferences, and audio mix settings.
5. Wait for audio extraction, speech recognition (unless a timed script is supplied), speaker detection, and translation.
6. Review the Georgian lines and detected speakers. Listen to the original speaker samples, inspect representative video frames, correct speaker genders, and assign available Georgian voices.
7. Continue to voice generation, background handling, synchronization, audio mixing, and MP4 export.
8. Download the completed MP4 from the project.

The project list supports reopening jobs. The online interface provides pause/resume, stop while preserving the project checkpoint, and project deletion. Deleting an active project requests cancellation; the project disappears from the list immediately, and its files are cleaned up after the active worker has stopped safely.

### Online source formats

Video uploads support:

- MP4
- MKV
- AVI
- MOV
- WEBM

The default maximum video size is 8 GB. It can be changed with `GEODUBBER_MAX_UPLOAD_BYTES`.

The online URL importer supports:

- A single YouTube video over HTTPS (not playlists).
- A direct public HTTPS URL ending in a supported video extension.
- A BitTorrent Magnet URI. GeoDubber selects one main supported video file from the torrent; it skips unrelated extras such as samples, subtitles and archives.

URL imports are limited by the same default 8 GB maximum. Magnet downloads require working peers/seeders and public trackers. Direct HTTPS imports reject private/local network destinations. Only import videos that you own or are authorized to process.

### Progress and downloads

The online UI shows overall and per-stage progress. An estimated remaining time is shown when enough timing data is available; it is an estimate, not a guarantee.

The MP4 download uses a short-lived download ticket so the browser can stream the file without first buffering the entire video in JavaScript memory. If a download link expires, press the download button again to create a new one.

## Input files and subtitles

### Importing a timed script

An SRT, VTT or TXT file can be uploaded as the project's timed dialogue script. Upload it before the video file.

- If the imported text is already Georgian, translation can be skipped and the text is used for voice generation.
- If the imported text is in another language, it is used as the source script for Georgian translation.
- Imported timecodes guide the placement of the generated Georgian dialogue.

The online API accepts script files up to 20 MB.

### Subtitle output is different from script input

For the **online profile**, subtitle files are input data only. They are not burned into the video, are not added as a selectable subtitle track, and separate SRT files are not exported. The online final MP4 contains the video and the Georgian dubbed audio without visible captions.

The **desktop profile** has its own subtitle settings. Depending on those settings, it can export SRT files and add a selectable or burned subtitle track.

## Windows desktop installation

### Requirements

- Windows 10 or Windows 11.
- Python 3.11.
- FFmpeg and FFprobe.
- An internet connection for package/model downloads and any configured online AI services.
- Enough free disk space for videos, models, temporary audio, and output files.

An NVIDIA GPU is optional. The installer checks for NVIDIA tooling and installs a CUDA-enabled PyTorch build when it detects an NVIDIA GPU; otherwise it installs the CPU build. CPU processing can be considerably slower.

### Install

1. Install Python 3.11 from [python.org](https://www.python.org/downloads/) and enable **Add Python to PATH**.
2. Extract or clone the repository.
3. Run `install.bat`.
4. If FFmpeg is still not found, install a Windows FFmpeg build and add `ffmpeg.exe` and `ffprobe.exe` to your PATH, or place them in `tools\ffmpeg\bin\`.
5. Run `start.bat`.

### First desktop run

1. Open **Settings → Models** and download the models needed by your selected ASR, translation, speaker-analysis, and TTS engines. Model weights are not bundled in the slim source distribution.
2. Configure the Gemini key through the `GEMINI_API_KEY` environment variable if Gemini features are required.
3. If using a gated Hugging Face speaker model, accept the model provider's terms and configure a read token in the appropriate local settings.
4. Open **Settings → Voices** and configure the Georgian voices required by your selected TTS engine. For reference-voice engines, use audio samples and transcripts that you have permission to use.
5. Load a short test video, run the pipeline, and listen to the result before processing a full-length film.
6. Check the Review stage and correct translation, speaker gender, and voice assignment before continuing.

The desktop output directory is configurable. The default project output folder is `output\`.

### Desktop speed presets

The bundled desktop presets adjust recognition model size, Whisper beam size, TTS concurrency, and video encoding settings.

| Preset | ASR model | Whisper beam | TTS workers |
|---|---:|---:|---:|
| Fast | tiny | 1 | 8 |
| Balanced | base | 1 | 6 |
| Quality | small | 5 | 4 |

Actual speed depends on the computer, video length, selected models, network/API response times, and whether background separation is enabled.

## Run the online API on Windows

This mode runs the browser interface and API on a Windows computer. It is useful for a personal processing host. Keep the API bound to localhost and expose it remotely only through an HTTPS reverse proxy or a trusted private tunnel.

### 1. Create the online virtual environment

Run these commands from the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
```

### 2. Install CPU PyTorch and online requirements

For a CPU-only Windows host:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-online.txt
```

If the machine has a supported NVIDIA setup, install the matching PyTorch build from the official PyTorch instructions instead of the CPU wheel. PyTorch is installed separately because the correct wheel depends on the host.

Install FFmpeg and FFprobe and make sure both commands are available in PowerShell:

```powershell
ffmpeg -version
ffprobe -version
```

### 3. Configure the online service

Set a persistent data folder and the required API token. The token must be at least 32 characters long. Set Gemini if you want Gemini translation and speaker analysis. HF_TOKEN is optional and is used only when a gated Hugging Face model is configured.

For the current PowerShell session, set the values as follows:

```powershell
$env:GEODUBBER_HOME = "$PWD\data"
$env:GEODUBBER_API_TOKEN = "replace-with-a-long-random-secret"
$env:GEMINI_API_KEY = "your-gemini-key"
# Optional:
$env:HF_TOKEN = "your-hugging-face-read-token"
```

For a persistent Windows user environment, use Windows Environment Variables or PowerShell's `[Environment]::SetEnvironmentVariable`. After changing persistent environment variables, restart the terminal or the background service so it receives the updated values.

Never commit secrets, paste them into source code, or put them in this README.

### 4. Start and verify the API

Start the service from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn online_api:app --host 127.0.0.1 --port 10000
```

Open the browser interface at:

```text
http://127.0.0.1:10000/
```

Check basic health:

```powershell
Invoke-RestMethod "http://127.0.0.1:10000/healthz"
```

The health endpoint does not require the API token. The authenticated status endpoint does:

```powershell
$token = $env:GEODUBBER_API_TOKEN
Invoke-RestMethod "http://127.0.0.1:10000/v1/status" -Headers @{ Authorization = "Bearer $token" }
```

Expected health response:

```json
{"status":"ok","service":"geodubber-api"}
```

For remote access, use HTTPS through a reverse proxy or a trusted private tunnel. Do not expose the raw API port 10000 directly to the public internet.

## Connect the Admin Panel

The Admin Panel is a separate application and can remain hosted on Render. GeoDubber's online API runs on the processing host. Configure the Admin Panel service with these environment variables:

| Variable | Value |
|---|---|
| `GEODUBBER_API_URL` | The HTTPS base URL of the online GeoDubber API, without a trailing slash |
| `GEODUBBER_API_TOKEN` | The same secret token configured on the GeoDubber API host |

After saving the environment variables, allow the Admin Panel service to redeploy. The Admin Panel URL does not move when the processing host changes; only its GeoDubber API URL needs to be updated.

Do not place the API token in frontend source files, public repositories, or browser URLs. The online API uses Bearer-token authentication for its protected endpoints.

## Deploy on Oracle Cloud

A Docker Compose deployment guide for an Oracle Cloud Always Free Ampere ARM VM is maintained separately:

[Oracle Cloud deployment guide](deploy/oracle-cloud/README.md)

That deployment uses the included `Dockerfile`, `docker-compose.yml` and Caddy reverse-proxy configuration. It keeps the Admin Panel on Render and runs the GeoDubber API on the VM. Oracle Always Free capacity and usage policies can change, and free resources do not provide a guaranteed uptime service.

## Configuration and secrets

### Online API environment variables

| Variable | Purpose |
|---|---|
| `GEODUBBER_API_TOKEN` | Required Bearer token; use a random secret of at least 32 characters |
| `GEODUBBER_HOME` | Root folder for online projects, cache and persistent data |
| `GEMINI_API_KEY` | Optional Gemini credential for translation and speaker analysis |
| `HF_TOKEN` | Optional Hugging Face read token for gated model access |
| `GEODUBBER_MAX_UPLOAD_BYTES` | Maximum video upload/import size in bytes; default is 8 GiB |
| `GEODUBBER_TORRENT_TIMEOUT` | Optional maximum torrent download time in seconds |

The online service uses CPU processing and a single worker for video jobs. Do not assume that adding API workers will increase throughput safely.

### Desktop credentials

Desktop Gemini access is read from the process environment variable `GEMINI_API_KEY`. The key is not intended to be stored in `settings.json`.

On Windows, you can set a persistent user variable from Command Prompt:

```cmd
setx GEMINI_API_KEY "YOUR_NEW_KEY"
```

Close and reopen the application after setting it. If a key has ever been exposed in a public repository or shared log, revoke it with the provider and create a replacement.

## Storage and backups

For the online API, project metadata, uploads, checkpoints and outputs are stored under `GEODUBBER_HOME`. The TTS cache is stored below the same data root. Choose a location with enough free disk space and back it up independently.

For the Oracle Docker deployment, the host directory `/var/lib/geodubber` is mounted into the container as persistent storage. Back it up outside the VM. Do not treat a free-tier VM as the only copy of important media.

For the desktop profile, protect the application data, model cache, projects, voices, and output folders before reinstalling or moving to another computer.

## Troubleshooting

### API returns 401 Unauthorized

- Check that the browser and Admin Panel use the same `GEODUBBER_API_TOKEN` as the API host.
- Confirm that the service process received the current environment variables.
- Restart the API after changing the token.
- Do not share the token in screenshots or support messages.

### API returns 503 because the token is not configured

The API token is missing or shorter than 32 characters in the running process. Set `GEODUBBER_API_TOKEN` and restart the service.

### Browser interface opens but cannot reach the API

- Confirm that `/healthz` works on the processing host.
- Confirm that the HTTPS tunnel or reverse proxy points to `http://127.0.0.1:10000` (or the internal API address in the Docker deployment).
- Confirm that the Admin Panel's `GEODUBBER_API_URL` has the right HTTPS base URL and no trailing slash.
- Check the tunnel/proxy status and Windows firewall rules.

### FFmpeg or FFprobe not found

Install FFmpeg and add its `bin` directory to PATH. Open a new terminal and verify with `ffmpeg -version` and `ffprobe -version`.

### Translation stays at 0% or is slow

- Check that `GEMINI_API_KEY` is configured in the process running the API.
- Review the server log for Gemini quota (429), temporary service (503/504), authentication, or model-not-found errors.
- Large batches, long films, API quotas, network latency, and retries can increase processing time.
- If a quota is exhausted, wait for the provider's reset window before retrying. Do not repeatedly restart a job while a request is still running.

### Some original dialogue remains audible

Demucs must be installed and usable for dialogue/background separation. If separation is unavailable or fails, the online pipeline keeps the original soundtrack quietly under the Georgian dub. Some original speech can therefore remain audible.

### Lines overlap or sound rushed

Review warnings for lines that exceed their available time window. Shorten the Georgian text in Review where possible. Generated speech can be sped up within configured limits, but a line may still overlap the next one if it is too long.

### Magnet import does not start or stalls

A valid Magnet link needs an info hash and reachable peers/seeders. Torrent metadata discovery can take several minutes. Only the selected supported video file is downloaded. Check the server log, tracker availability, disk space, and the configured torrent timeout.

### Video download fails

- For YouTube, use one HTTPS video URL, not a playlist.
- For direct links, use a public HTTPS URL that ends in MP4, MKV, AVI, MOV or WEBM.
- Private/local network URLs and non-HTTPS direct links are rejected.
- Confirm that the source can be accessed by the processing host, not just by your browser.

## Tests and project layout

Run the automated tests from the repository root:

```powershell
python -m pytest tests
```

The tests cover core processing logic, media handling, synchronization, voice assignment, settings, and online API behavior. Many tests use stand-in model backends; passing the test suite does not by itself verify the quality or availability of real Gemini, Whisper, TTS, Demucs, YouTube or torrent services on a particular computer.

Important project paths:

| Path | Purpose |
|---|---|
| `app.py` | Desktop application entry point |
| `online_api.py` | FastAPI service and browser API |
| `web/index.html` | Browser interface |
| `config/` | Application settings and configuration |
| `core/` | Processing pipeline, translation, audio, video and import logic |
| `scripts/` | Maintenance and verification scripts |
| `tests/` | Automated tests |
| `deploy/oracle-cloud/` | Oracle Cloud Compose and HTTPS deployment files |
| `requirements.txt` | Desktop runtime dependencies |
| `requirements-online.txt` | Online API runtime dependencies |
| `install.bat` / `start.bat` | Windows desktop installation and startup |

## Known limitations

- The online API processes one video job at a time.
- The online runtime is CPU-oriented. Long videos can take a long time, especially on low-power computers.
- Gemini is subject to provider availability, rate limits, quotas and model changes.
- Speaker count and perceived gender are estimates. Listen to the original samples and confirm them in Review.
- Demucs separation may be unavailable or fail on a particular host; the fallback can leave faint original speech under the dub.
- Timing fit is constrained by the available speech window. Some long Georgian lines may still overlap.
- The automated test suite does not replace listening to the completed dub or testing the real integrations on the target machine.
- Dependency requirements use version ranges rather than a fully pinned, reproducible lock file. A clean installation may resolve newer package versions.

Before processing a full-length video, test a short clip, review the translation and voices, and listen to the final MP4.
