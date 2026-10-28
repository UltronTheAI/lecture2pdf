# Lecture2PDF

Installable Python 3 CLI to extract lecture slides, fuse unobstructed temporal pixels, run OCR and render a searchable study PDF.

## Setup

Python 3.10+ and Tesseract OCR are required for text extraction. On Windows: install Tesseract and add its installation directory to PATH. FFmpeg-supported video codecs are required by your OpenCV build.

```powershell
cd lecture2pdf
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
pytest -q
```

Linux: `sudo apt install tesseract-ocr` and use `source .venv/bin/activate`.

## Phase 2: Run

```powershell
lecture2pdf "C:\videos\lecture.mp4" -o "C:\videos\lecture_notes.pdf" --quality maximum --mode study --compute auto --decoder auto --ocr-engine auto --reconstruction temporal --roi "0.03,0.05,0.94,0.82" --verbose
```

`--compute gpu` is strict: it performs a runtime CUDA allocation probe and fails if no operational CUDA provider is available. `--compute auto` selects CUDA only when that probe succeeds; otherwise it uses CPU.

`--roi` is strongly recommended to exclude player chrome and borders. ROI is normalized `x,y,width,height` relative to the input frame. For no crop, omit it.

Outputs: `lecture_notes.pdf`, `lecture_notes_assets/slide_0001.png`, and `lecture_notes_assets/manifest.json` with OCR and timestamps.

## Architecture

- `video/`: FFprobe metadata, bounded timestamp sampling, FFmpeg/OpenCV decoder adapters, and NVDEC feature detection
- `detect.py`: SSIM/edge/pixel change scoring, perceptual-hash dedup, stable transitions
- `reconstruct.py`: translation alignment and temporal median across stable-slide frames
- `ocr.py`: CLAHE-enhanced Tesseract OCR
- `pdf.py`: ReportLab pages with searchable text beneath original/reconstructed images
- `cli.py`: command line and orchestration
- `tests/`: synthetic detection, reconstruction, PDF tests

## Known limitations

This is a production-oriented v0.2 engine, but not every optional capability is available on every machine. Presenter segmentation is not bundled; `temporal-segmentation` currently falls back to conservative temporal reconstruction unless an external model integration is added. A stationary presenter cannot be removed without inventing pixels. Slide animation and projector flicker still require lecture-specific threshold calibration. OCR quality depends on the installed engine and does not claim perfect mathematical notation. No speech transcription or AI note generation is included. CUDA and NVDEC are optional and are reported unavailable when runtime probes fail.

Recommended next milestones: persistent frame cache with resumable SQLite jobs; slide boundary refinement and animation merge; foreground masks with segmentation; CUDA-backed OCR; memory-bounded streaming PDF; video regression fixtures; confidence and provenance labels; native PDF slide reconstruction from text/layout.

## Lecture2PDF v0.2

The current implementation is v0.2.0. It adds a truthful acceleration layer, FFmpeg/FFprobe-aware bounded video decoding, adaptive multi-signal detection, conservative observed-pixel reconstruction, progress/ETA reporting, SQLite checkpoints, cache fingerprints, OCR caching, quality presets, and richer searchable PDFs. See [docs/ARCHITECTURE_AUDIT.md](docs/ARCHITECTURE_AUDIT.md) for the migration record.

### Windows PowerShell setup

```powershell
cd C:\path\to\lecture2pdf
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Install FFmpeg and ensure `ffmpeg.exe` and `ffprobe.exe` are on PATH. Install Tesseract OCR separately and ensure `tesseract.exe` is on PATH for searchable text. The `paddle` and `gpu` extras are optional; use a CUDA-compatible PyTorch wheel from the PyTorch installation selector rather than assuming the generic package matches the installed NVIDIA driver.

### Usage

```powershell
lecture2pdf "C:\videos\lecture.mp4" `
  --output "C:\videos\notes.pdf" `
  --quality maximum `
  --mode study `
  --compute auto `
  --decoder auto `
  --ocr-engine auto `
  --reconstruction temporal `
  --sample-fps 2 `
  --resume `
  --verbose
```

`--compute auto` selects CUDA only after a real provider allocation and synchronization probe. `--compute gpu` is strict and exits with an actionable error if CuPy/PyTorch CUDA is not operational. The present environment has an RTX 5050 but the installed Python stack does not expose an operational CUDA provider, so CPU is the measured fallback; no GPU speedup is claimed. `--decoder nvdec` is also strict and requires FFmpeg to advertise a compatible CUVID/CUDA decoder. Auto mode does not assume hardware decoding is faster.

Quality presets are measurable cost/quality defaults: `fast` uses 1 FPS, one stability sample, and a nine-frame fusion bound; `balanced` uses 1.5 FPS and moderate fusion; `maximum` uses 2 FPS, two stability samples, and a 31-frame fusion bound. Explicit flags override preset values.

### Cache and resume

Jobs are stored under `.lecture2pdf-cache/jobs/<fingerprint>/` by default, with `metadata.sqlite`, `slides/`, `reconstructed/`, `ocr/`, and checkpoint data. `--cache-dir` changes the root. `--resume` validates the input/configuration fingerprints and reuses completed slide artifacts. `--clear-cache` removes the exact current job cache before starting. Ctrl+C writes a checkpoint when persistence succeeds and prints a resume command; a failed persistence operation is reported as an error.

### Troubleshooting

* GPU requested but unavailable: run `lecture2pdf ... --compute auto` or `--compute cpu`, or install a CUDA-enabled PyTorch/CuPy build compatible with the NVIDIA driver. `opencv-python-headless` from PyPI is not assumed to contain CUDA.
* NVDEC unavailable: use `--decoder auto` or `--decoder opencv`. FFmpeg build support is feature-detected at runtime; a visible NVIDIA adapter alone is not sufficient.
* OCR unavailable: install Tesseract and its English language data, or use `--no-ocr` for image-only PDFs. PaddleOCR is optional and never silently replaces an explicitly requested Tesseract engine.
* Large memory use: lower `--max-fusion-frames`, choose `--quality fast`, use an ROI, and retain the disk cache. The pipeline never loads the entire video into RAM.

### Cleanup

Remove caches, Python bytecode, test caches, build output, and benchmark output from the project directory:

```powershell
lecture2pdf --cleanup
```

To also remove generated `*_assets` directories and local `.log`/`.tmp` files:

```powershell
lecture2pdf --cleanup-all
```

Cleanup preserves source videos and PDFs. Use `--cleanup-root "C:\path\to\project"` to clean another project directory.

### Benchmarks

The scripts in `benchmarks/` print measured JSON for decode, detection, reconstruction, and end-to-end conversion. Run the CPU and CUDA end-to-end comparison only on the same input and equivalent flags. If CUDA is unavailable, the result says `unavailable`; no benchmark result is fabricated.
