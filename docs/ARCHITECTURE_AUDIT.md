# Lecture2PDF Architecture Audit

**Audit date:** 2026-10-28  
**Repository:** `Lecture2PDF v0.1`  
**Scope:** Read-only audit of the existing package, CLI, tests, declared dependencies, and available media. No source code or configuration was changed as part of this audit.

## Executive summary

Lecture2PDF is a functional, synchronous v0.1 prototype that converts locally stored lecture video into a searchable ReportLab PDF. The current path is:

`CLI -> OpenCV VideoCapture -> sampled frames -> multi-signal slide detector -> temporal reconstruction -> Tesseract OCR -> ReportLab PDF + JSON manifest`

The prototype already has useful foundations: sequential bounded video decoding, configurable sampling, normalized rectangular ROI support, SSIM/edge/pixel/hash-based comparison, translation alignment, temporal median reconstruction, CLAHE OCR preprocessing, searchable PDF output, timestamps, and a manifest. Existing tests pass.

It is not yet a CUDA-enabled production engine. `--compute gpu` does not execute CUDA work and currently continues on CPU with a warning. There is no `auto` compute mode, no decoder abstraction, no NVDEC integration, no GPU memory budget, no checkpoint database, no resume path, no progress/ETA system, no benchmark harness, and no OCR cache. The current implementation also retains slide/frame data in memory longer than necessary and cannot provide reliable crash recovery.

The recommended migration is incremental: first establish explicit runtime capability detection and a CPU-compatible backend contract, then add measurable GPU acceleration to batchable image operations, then introduce a bounded decoder/processing pipeline, persistent checkpoints, richer observability, and only afterward optional segmentation or GPU OCR. CPU behavior and the existing CLI should remain the reference implementation throughout.

## Audit evidence

### Repository contents inspected

- `pyproject.toml`
- `README.md`
- `lecture2pdf/cli.py`
- `lecture2pdf/video.py`
- `lecture2pdf/detect.py`
- `lecture2pdf/reconstruct.py`
- `lecture2pdf/ocr.py`
- `lecture2pdf/pdf.py`
- `lecture2pdf/__init__.py`
- `tests/test_pipeline.py`
- `download.py`
- available files under `downloads/`, `notes_assets/`, and `vocabulary_notes_assets/`

No `docs/` or `benchmarks/` directory existed before this audit.

### Runtime and dependency observations

The inspected environment reported:

- Python: Anaconda Python 3.14.3
- OpenCV: 5.0.0
- NumPy: 2.3.5
- scikit-image: 0.25.2
- ReportLab: 5.1.0
- FFmpeg: 9.0.1 installed on the system
- NVIDIA GPU: RTX 5050 Laptop GPU, approximately 8 GB VRAM
- OpenCV CUDA device count: `0`
- Tesseract executable: not installed or not available on `PATH`

The project declares an optional PaddleOCR dependency, but the current implementation does not use it.

### Existing validation

The existing test suite contains five tests and passed with:

```text
5 passed in 1.36 seconds
```

This is useful baseline evidence for the prototype, but it is not an integration, performance, CUDA, resume, or PDF-searchability test suite.

### Available media

Two representative videos are present under `downloads/`:

| File | Size | Container / video | Dimensions | Nominal rate | Duration |
|---|---:|---|---:|---:|---:|
| `Communication and Communication Processes … [5WaAWU4ef_s].mp4` | 72,173,001 bytes | MP4 / AV1 | 1280×720 | 30 FPS | 1,223.93 s |
| `Vocabulary … Lecture 05 … [oEa7UeS9AaY].mp4` | 47,796,873 bytes | MP4 / AV1 | 1920×1080 | 30 FPS | 540.83 s |

These files are suitable for later end-to-end and benchmark work. The AV1 streams are particularly relevant when evaluating decoder compatibility and whether hardware decode is actually available in the installed FFmpeg build.

## Existing architecture

### CLI and entry point

`lecture2pdf/cli.py` is the orchestration layer. It validates the input path and current options, creates an output-specific asset directory, invokes the video/detection/reconstruction/OCR/PDF stages, and writes a JSON manifest alongside the generated PDF.

The current interface supports local video input, output selection, quality/mode arguments, sample rate and ROI-related settings, `--verbose`, and disabling OCR. The accepted behavior is limited: only `--quality maximum` and `--mode study` are implemented, and the current compute behavior is CPU-only even when the user requests `gpu`.

### Video decoding and sampling

`lecture2pdf/video.py` uses OpenCV `VideoCapture` and sequential decoding. Frames are sampled approximately by timestamps. The implementation does not use an FFmpeg subprocess, PyAV, timestamp-aware packet/frame handling, NVDEC, a decoder interface, a bounded producer/consumer queue, or a separate metadata model.

Sequential decoding is a reasonable prototype choice for sparse sampling because repeated random seeks can be slower and less accurate on compressed video. However, the current implementation does not explicitly compare sequential decoding with timestamp seeking, does not expose decoder capability information, and does not distinguish decode failures from end-of-stream conditions with production-grade diagnostics.

### Slide detection

`lecture2pdf/detect.py` contains `SlideDetector`. The current comparison path combines several CPU-side signals:

- SSIM-like structural comparison
- Canny edge differences
- changed-pixel ratio
- perceptual hashing

The detector confirms changes across stable samples and performs duplicate rejection. This is stronger than a single raw pixel threshold, but thresholds are fixed and the detector does not yet include adaptive recent-frame statistics, histogram signals, text-region changes, temporal stability scoring, candidate-versus-verification stages, or explainable confidence metadata.

There is no explicit classification of a transition as a new slide, animation, presenter movement, camera movement, duplicate slide, or incremental whiteboard update.

### Reconstruction

`lecture2pdf/reconstruct.py` performs translation-only ECC alignment and applies a temporal median over frames. This provides a useful baseline for reducing transient foreground obstruction, but it has important correctness and coverage limits:

- no presenter segmentation mask;
- no geometric model beyond translation;
- no explicit motion-blur rejection;
- no mask-aware unobstructed-region selection;
- no confidence estimate;
- no preservation of an original representative frame as a first-class reconstruction artifact;
- no explicit uncertainty representation for permanently occluded content.

Temporal median reconstruction must remain conservative: it can combine observed pixels, but it must not invent hidden academic content.

### OCR

`lecture2pdf/ocr.py` uses Tesseract with CLAHE-style preprocessing. The current output path retains text but discards OCR bounding boxes and confidence information. There is no reading-order model, OCR cache, duplicate-text removal, selectable OCR engine, or GPU OCR implementation. PaddleOCR is declared as optional but is not wired into the application.

Because Tesseract is not installed in the inspected environment, OCR-enabled end-to-end execution cannot currently be considered portable or verified on this machine. The existing `--no-ocr` path is therefore important for baseline pipeline validation.

### PDF generation

`lecture2pdf/pdf.py` uses ReportLab to generate an A4 searchable PDF with slide images, timestamps, and OCR text. The image and text are associated with the same slide, and the existing output preserves a searchable-document workflow.

The current PDF layer does not provide a table of contents, richer metadata, explicit page numbering, robust Unicode font selection, configurable compression, or a documented distinction between visual slide fidelity and OCR fidelity. It should remain CPU-based; PDF serialization is not a worthwhile first CUDA target.

## Current execution path and data flow

1. The CLI validates the input and creates an output asset directory.
2. OpenCV opens the video and decodes frames sequentially.
3. Frames near configured sample timestamps are passed into slide detection.
4. The detector compares each sampled frame against recent state and accepts or rejects slide transitions.
5. Accepted slide segments retain frame references for reconstruction.
6. Reconstruction aligns selected frames and computes a temporal median.
7. OCR preprocesses the reconstructed/selected slide and invokes Tesseract.
8. ReportLab lays out the slide image and searchable OCR text.
9. A JSON manifest is written beside the PDF.

The path is synchronous and single-threaded. There is no independently bounded queue between decode and analysis, no asynchronous stage overlap, and no durable state boundary between stages.

## Performance bottlenecks

### Decode and sampling

OpenCV decoding is CPU-bound in the current application. Every decoded frame is handled in a Python-controlled loop, and the application has no direct access to FFmpeg hardware-decoder capability reporting. The two available inputs use AV1, so codec support and hardware-path availability must be measured rather than inferred from the presence of an NVIDIA GPU.

The current sampler is timestamp-approximate and does not yet make an explicit sequential-versus-seek decision. Sparse seeking can cause redundant keyframe decoding, while sequential decoding can decode many frames that will not be analyzed. The correct choice depends on codec, GOP structure, sample density, and the actual FFmpeg/OpenCV build.

### CPU image comparisons

SSIM, edge extraction, changed-pixel calculations, hashing, and repeated resizing/color conversion are CPU-side operations. These are data-parallel and may benefit from batching, but individual small-frame transfers to a GPU can cost more than the computation. A GPU path should therefore operate on batches of same-shaped frames and return only compact comparison results where possible.

### Reconstruction memory and computation

Alignment and temporal median operations are image-wide and can be expensive at 1280×720 or 1920×1080. Holding many frames or slide segments in Python memory increases peak RAM and makes long lectures less predictable. Reconstruction is a good candidate for bounded, VRAM-aware batches after profiling, but it should not move whole videos or all slide frames to device memory.

### OCR

Tesseract is CPU-bound and can dominate end-to-end processing when many unique slides are accepted. OCR is also a poor candidate for unconditional per-sample execution. OCR should be applied selectively to accepted slide representatives/reconstructions, cached by content hash, and instrumented separately from decode and detection time.

### PDF serialization

ReportLab generation is expected to remain CPU-bound and should be treated as a finalization stage. It is not a useful target for CUDA acceleration. Its performance impact should be measured, but it should not complicate the GPU architecture.

### Observability overhead and perceived stalls

The current application emits sparse slide-detection logs and has no stage progress, sampled-frame count, processing rate, memory telemetry, or ETA. This makes long CPU intervals appear frozen even when work is progressing. The absence of progress is a reliability and usability problem, not merely a presentation issue.

## GPU opportunity assessment

### What is currently proven

The machine has an RTX 5050 Laptop GPU, but OpenCV reports zero CUDA devices. Therefore the current repository has no verified usable CUDA execution path. The presence of a GPU alone is not evidence that `--compute gpu` can work.

FFmpeg 9.0.1 is installed and includes NVDEC/NVENC-related build support, but the application does not invoke FFmpeg and no runtime decoder capability test has been performed. NVDEC must be verified with the actual codec, driver, FFmpeg build, and selected hardware device before being advertised.

### Recommended first GPU workloads

The most maintainable initial GPU scope is batched image processing:

- resize and normalization of sampled frames;
- BGR/RGB and grayscale conversion;
- frame differencing and compact similarity statistics;
- edge-map generation if profiling shows it is worthwhile;
- batched histogram or hash calculation;
- temporal fusion for bounded frame batches.

These workloads have regular tensor/array shapes and can be isolated behind a backend contract. Detection decisions, checkpoint transactions, OCR result handling, and PDF serialization should remain backend-independent and CPU-safe.

### Backend selection recommendation

Use one primary GPU array/tensor framework rather than introducing PyTorch, CuPy, OpenCV CUDA, and ONNX Runtime simultaneously. The choice should be based on measured Windows compatibility, install footprint, supported Python version, and the amount of data that can remain on device.

A modular backend should expose capability checks and operations such as `resize_batch`, `grayscale_batch`, `difference_batch`, and `fuse_batch`. It should report the selected device, framework, available memory, and unsupported operations. `auto` may select a verified CUDA backend; `cpu` must disable CUDA; explicit `gpu` must fail with an actionable error when no operational backend is available. It must never silently fall back to CPU for an explicit GPU request.

### Memory and transfer constraints

The target GPU has approximately 8 GB VRAM, but the application must reserve headroom for the driver, display, runtime, and other processes. Batch sizes should be selected from a configurable memory budget and actual free memory, not from total VRAM. Pinned host buffers, asynchronous transfers, and CUDA streams should be introduced only after profiling confirms a benefit. Every benchmark involving asynchronous kernels must synchronize before timing.

### NVDEC assessment

NVDEC is a decoder capability, not a generic OpenCV feature. The migration should first add an FFmpeg-based decoder probe that reports available hardware accelerators, codec support, selected device, and fallback reason. Hardware decode should be optional and measurable. If frames must be copied back to host memory for CPU detection/OCR, end-to-end speed may be worse than high-quality CPU decoding; the benchmark must report this rather than assuming NVDEC is faster.

## Reliability and correctness issues

1. **Incorrect GPU contract:** explicit `--compute gpu` currently does not guarantee CUDA and may continue on CPU.
2. **No explicit auto mode:** runtime capability selection is not represented in the CLI or configuration model.
3. **No durable progress state:** process termination loses the current job position and completed work.
4. **No resume validation:** there is no deterministic job ID, video fingerprint, configuration fingerprint, or cache invalidation policy.
5. **Unbounded retention risk:** slide/frame references remain in memory until completion, which is unsafe for long or high-resolution inputs.
6. **Sparse failure diagnostics:** decoder, OCR, and stage failures do not have a durable, stage-specific recovery model.
7. **OCR environment dependency:** Tesseract is not available on the audited machine, so OCR failure is likely unless the user installs it or disables OCR.
8. **Weak transition semantics:** fixed thresholds can mistake presenter movement, camera movement, flicker, animation, or obstruction for a new slide.
9. **Reconstruction uncertainty is implicit:** temporal median may not distinguish observed content from permanently hidden content and has no confidence output.
10. **VFR/timestamp limitations:** the current approximate timestamp sampling has no explicit variable-frame-rate model or presentation-timestamp abstraction.
11. **No bounded producer/consumer design:** decode and processing cannot apply backpressure independently.
12. **Limited output metadata:** PDFs lack a table of contents and richer document metadata, and OCR confidence is not preserved.
13. **Insufficient test coverage:** the current five passing tests do not exercise real video, VFR input, interruptions, resume, GPU capability, OCR correctness, or searchable PDF output.

## Proposed migration strategy

The migration should preserve the current CPU pipeline as the reference behavior and add capability behind stable interfaces.

### Stage 1: establish contracts and observability

- Add typed configuration validation for compute, decoder, OCR, reconstruction, quality, sampling, cache, and logging options.
- Introduce a device/capability report with `auto`, `cpu`, and strict `gpu` semantics.
- Add structured stage timing, progress counters, measured processing FPS, ETA, logging, and RAM/VRAM telemetry.
- Keep the current OpenCV decoder and Tesseract path as the CPU baseline.
- Record a reproducible baseline on both available videos, using OCR-disabled runs if Tesseract is unavailable.

### Stage 2: isolate video and sampling

- Add `lecture2pdf/video/` with metadata, sampling, decoder interfaces, and an FFmpeg implementation.
- Preserve sequential decoding as the fallback.
- Add capability-tested FFmpeg hardware decode, including codec/device verification and clean subprocess shutdown.
- Model presentation timestamps explicitly and support bounded frame queues/backpressure.
- Benchmark sequential versus seek-based sampling before selecting a default.

### Stage 3: add a real acceleration backend

- Add `lecture2pdf/acceleration/` with device selection, CPU backend, CUDA backend, memory budgeting, and optional profiling.
- Choose a single primary framework after compatibility testing on Windows and CPU-only systems.
- Move only profiled, batchable image operations to the GPU.
- Keep CPU fallback for unsupported operations, but reject explicit `gpu` when the requested backend is unavailable.
- Add synchronization-aware benchmarks that separate initialization, transfer, kernel, and end-to-end times.

### Stage 4: improve detection and reconstruction

- Use inexpensive candidate signals first, followed by expensive verification only at candidate transitions.
- Add adaptive thresholds based on recent signal distributions and preserve explainable per-transition metadata.
- Add ROI-aware and text-region-aware comparisons.
- Add quality scoring and motion-blur rejection for selected frames.
- Make temporal reconstruction mask-aware and conservative; preserve representative and reconstructed images together and report confidence/uncertainty.
- Treat segmentation as optional and capability-gated. Do not make a model dependency mandatory for CPU installations.

### Stage 5: persistence and resume

- Add a deterministic job fingerprint based on input video fingerprint and relevant configuration.
- Use SQLite for job/stage metadata and atomic checkpoint transactions.
- Store frames, slides, reconstructions, and OCR outputs in a disk-backed cache with content hashes.
- On interruption, stop intake, persist a verified checkpoint, terminate decoding, release resources, and print a resumable command.
- Validate the input and configuration fingerprints before resuming.

### Stage 6: OCR and PDF quality

- Preserve Tesseract as the tested baseline and add explicit `auto`, `tesseract`, and optional `paddle` selection.
- Cache OCR by image/content hash and retain confidence plus bounding boxes.
- Add reading-order reconstruction and duplicate text suppression without replacing OCR with generated text.
- Improve PDF metadata, page numbering, table of contents, Unicode font handling, and compression while retaining the original slide image.

### Stage 7: tests and benchmarks

- Add synthetic lecture-video fixtures with known transitions, duplicates, moving foregrounds, incremental content, and VFR timing.
- Add unit tests for detector signals, adaptive thresholds, reconstruction, OCR normalization, cache invalidation, and checkpoint recovery.
- Add integration tests for CPU-only conversion, strict GPU failure/success behavior, interruption/resume, and searchable PDFs.
- Add benchmark scripts for decode, detection, reconstruction, and end-to-end conversion.
- Report measured throughput, total time, peak RAM, peak VRAM, initialization overhead, transfer overhead, and quality metrics. Do not report an acceleration factor until measured on the same input and equivalent settings.

## Recommended target architecture

```text
CLI/config validation
        |
Capability report + selected runtime
        |
Decoder -> bounded sample queue -> batched preprocessor
                                      |
                         CPU or verified CUDA backend
                                      |
Candidate detector -> transition verifier -> segment store
                                      |
                 frame selector -> optional reconstruction
                                      |
                         OCR cache / OCR engine
                                      |
                 PDF renderer + manifest + job checkpoint
```

Cross-cutting services should include structured logging, progress/ETA reporting, memory telemetry, cancellation handling, atomic cache writes, and a job state database. Each stage should be restartable from durable artifacts rather than depending on Python object state from an earlier stage.

## Baseline and validation gaps

The audit verified the existing five-test suite and the environment/media inventory. It did not establish a trustworthy end-to-end throughput baseline because the repository has no benchmark harness and Tesseract is unavailable in the inspected environment. No CUDA workload was measured because OpenCV reported no CUDA device and the application has no alternate CUDA backend.

Before claiming v0.2 performance improvements, record at minimum:

- decode FPS and sampled-frame rate;
- detector FPS;
- reconstruction time per slide;
- OCR time per slide and total OCR time;
- end-to-end wall time;
- peak RAM;
- CUDA initialization and transfer overhead;
- peak VRAM and GPU utilization when CUDA is operational;
- slide precision/recall and duplicate rejection accuracy on synthetic ground truth;
- OCR character error rate where ground truth exists.

## Audit conclusion

The v0.1 codebase is a viable small CPU prototype and should be evolved rather than replaced wholesale. Its strongest migration assets are the existing modular stage files, sequential decode loop, multi-signal detector baseline, temporal reconstruction baseline, searchable PDF path, and passing tests. Its most urgent production blockers are the false GPU contract, absent capability detection, lack of persistence/resume, lack of bounded resource management, weak observability, and limited integration/benchmark coverage.

The first implementation milestone should be a truthful, observable CPU/auto/GPU runtime contract plus reproducible baseline measurements. That creates the evidence needed to decide whether CUDA preprocessing, FFmpeg hardware decode, or GPU OCR improves end-to-end throughput on the actual AV1 lecture inputs, instead of optimizing isolated operations without a measured production benefit.

## Final v0.2 verification note

The installed optional CUDA provider passed a real allocation/synchronization probe on the RTX 5050 and the v0.2 detector uses a coarse CUDA-backed comparison signal in GPU mode. OpenCV CUDA still reports zero devices, Tesseract is still absent on PATH, and the benchmark shows CUDA slower than CPU for the small AV1 workload because decode remains software and transfer overhead dominates.
