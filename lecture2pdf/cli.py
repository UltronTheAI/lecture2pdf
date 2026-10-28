import argparse
import json
import logging
import sys
from pathlib import Path
import cv2
from .atomic import atomic_write_bytes, atomic_write_json
from .video import sample_video
from .video import probe_video
from .detect import SlideDetector, difference
from .reconstruct import reconstruct
from .ocr import extract
from .pdf import build_pdf
from .interrupts import InterruptedRun
from .job import JobContext

LOG = logging.getLogger("lecture2pdf")

def parse_roi(raw):
    if not raw:
        return None
    try:
        values = tuple(float(v) for v in raw.split(","))
        if len(values) != 4:
            raise ValueError()
        x, y, w, h = values
        if min(values) < 0 or w <= 0 or h <= 0 or x+w > 1 or y+h > 1:
            raise ValueError()
        return values
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be normalized x,y,w,h in [0,1]") from exc

def run(args):
    src = Path(args.video)
    if not src.is_file():
        raise FileNotFoundError(src)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = out.parent / (out.stem + "_assets")
    cache.mkdir(parents=True, exist_ok=True)
    if args.compute == "gpu":
        LOG.warning("GPU requested: core OpenCV decode/detection/reconstruction uses CPU; GPU acceleration is not implemented in v0.1. No GPU performance claim is made.")
    metadata = probe_video(src)
    config = {
        "output": str(out.resolve()),
        "quality": getattr(args, "quality", "maximum"),
        "mode": getattr(args, "mode", "study"),
        "compute": getattr(args, "compute", "cpu"),
        "decoder": getattr(args, "decoder", "auto"),
        "sample_fps": args.sample_fps,
        "change_threshold": args.change_threshold,
        "fusion_threshold": args.fusion_threshold,
        "stable_samples": args.stable_samples,
        "max_fusion_frames": args.max_fusion_frames,
        "roi": args.roi,
        "ocr_lang": args.ocr_lang,
        "no_ocr": args.no_ocr,
    }
    cache_dir = getattr(args, "cache_dir", None) or (out.parent / ".lecture2pdf-cache")
    with JobContext.open(
        src,
        config,
        cache_dir=cache_dir,
        total_duration=metadata.duration,
        resume=getattr(args, "resume", False),
        clear_cache=getattr(args, "clear_cache", False),
        stream=sys.stderr,
        quiet=getattr(args, "quiet", False),
        verbose=args.verbose,
        log_file=getattr(args, "log_file", None),
    ) as job:
        detector = SlideDetector(change_threshold=args.change_threshold, min_stable=args.stable_samples)
        slides = []
        window = []
        job.progress.metrics.stage = "detecting slides"
        # Bounded per-slide reservoir. Keep evenly spaced frames across each segment.
        for sample in sample_video(src, fps=args.sample_fps, roi=args.roi, decoder=getattr(args, "decoder", "auto")):
            job.interrupts.raise_if_requested()
            decision = detector.push(sample)
            if decision is not None:
                if slides:
                    slides[-1]["frames"] = window[:]
                timestamp, frame = decision
                slides.append({"time": timestamp, "frame": frame, "frames": []})
                window = []
                LOG.info("Unique slide %d at %.1fs", len(slides), timestamp)
            if slides:
                if len(window) < args.max_fusion_frames and difference(sample.frame, slides[-1]["frame"]) < args.fusion_threshold:
                    window.append(sample.frame.copy())
                elif len(window) >= args.max_fusion_frames:
                    # Reservoir of recent frames improves chances of uncovering moving presenter.
                    window.pop(0)
                    window.append(sample.frame.copy())
            metrics = job.progress.metrics
            metrics.update(
                processed_timestamp=sample.time,
                sampled_frames=metrics.sampled_frames + 1,
                unique_slides=len(slides),
                duplicates_rejected=max(0, metrics.sampled_frames - len(slides)),
            )
            job.progress.emit()
        if slides:
            slides[-1]["frames"] = window
        if not slides:
            raise RuntimeError("No decodable frames found")
        job.checkpoint(
            "detect",
            status="completed",
            processed_timestamp=job.progress.metrics.processed_timestamp,
            sampled_frames=job.progress.metrics.sampled_frames,
            slides=[{"time": slide["time"]} for slide in slides],
        )

        output_slides = []
        job.progress.metrics.stage = "reconstructing and OCR"
        for i, slide in enumerate(slides):
            job.interrupts.raise_if_requested()
            image_path = cache / f"slide_{i+1:04d}.png"
            text_path = job.cache.child("ocr", f"slide_{i+1:04d}.json")
            item_key = f"slide:{i + 1}:{slide['time']:.6f}"
            if getattr(args, "resume", False) and job.checkpoints.is_completed(item_key) and image_path.is_file():
                clean = cv2.imread(str(image_path))
                if clean is None:
                    raise OSError(f"Failed to read cached {image_path}")
                text = ""
                if text_path.is_file():
                    text = json.loads(text_path.read_text(encoding="utf-8")).get("text", "")
                output_slides.append({"time": slide["time"], "frame": clean, "text": text})
                LOG.info("Resumed slide %d/%d", i + 1, len(slides))
                continue
            frames = slide["frames"]
            clean = reconstruct(frames) if len(frames) >= 3 else slide["frame"]
            image_path.parent.mkdir(parents=True, exist_ok=True)
            encoded_ok, encoded = cv2.imencode(".png", clean)
            if not encoded_ok:
                raise OSError(f"Failed to write {image_path}")
            atomic_write_bytes(image_path, encoded.tobytes())
            text = "" if args.no_ocr else extract(clean, language=args.ocr_lang)
            job.cache.write_json("ocr", f"slide_{i+1:04d}.json", {"text": text})
            output_slides.append({"time": slide["time"], "frame": clean, "text": text})
            job.checkpoints.mark_completed(item_key, {"image": str(image_path), "ocr": str(text_path)})
            job.progress.metrics.update(unique_slides=i + 1)
            job.progress.emit()
            LOG.info("Processed slide %d/%d", i+1, len(slides))
        build_pdf(output_slides, out, title=args.title)
        manifest = [{"slide": i+1, "timestamp_s": s["time"], "image": f"slide_{i+1:04d}.png", "ocr_text": s["text"]} for i,s in enumerate(output_slides)]
        atomic_write_json(cache / "manifest.json", manifest)
        job.cache.write_json("slides", "manifest.json", manifest)
        job.checkpoint("pdf", status="completed", output=str(out), slides=len(output_slides))
        job.progress.metrics.stage = "completed"
        job.progress.emit(force=True)
        LOG.info("Wrote %s (%d slides)", out, len(slides))
        return out

def main(argv=None):
    p = argparse.ArgumentParser(prog="lecture2pdf", description="Extract lecture slides into a searchable study PDF")
    p.add_argument("video", help="Input MP4/MKV/MOV/WebM video")
    p.add_argument("-o", "--output", default="notes.pdf")
    p.add_argument("--title", default="Lecture Study Notes")
    p.add_argument("--quality", choices=["maximum"], default="maximum")
    p.add_argument("--mode", choices=["study"], default="study")
    p.add_argument("--compute", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--decoder", choices=["auto", "ffmpeg", "opencv"], default="auto", help="Video decoder backend; auto feature-detects FFmpeg")
    p.add_argument("--sample-fps", type=float, default=2.0)
    p.add_argument("--change-threshold", type=float, default=0.19)
    p.add_argument("--fusion-threshold", type=float, default=0.24)
    p.add_argument("--stable-samples", type=int, default=2)
    p.add_argument("--max-fusion-frames", type=int, default=31)
    p.add_argument("--roi", type=parse_roi, help="Crop normalized x,y,w,h e.g. 0.04,0.05,0.92,0.82")
    p.add_argument("--ocr-lang", default="eng")
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    p.add_argument("--log-file", type=Path, help="Write JSON progress snapshots to this file")
    p.add_argument("--resume", action="store_true", help="Resume completed detection/OCR work from the job cache")
    p.add_argument("--cache-dir", type=Path, default=None, help="Root directory for resumable job caches")
    p.add_argument("--keep-cache", action="store_true", help="Keep the cache after conversion")
    p.add_argument("--clear-cache", action="store_true", help="Clear this job cache before processing")
    args = p.parse_args(argv)
    p.error("--sample-fps, --stable-samples and --max-fusion-frames must be positive") if args.sample_fps <= 0 or args.stable_samples <= 0 or args.max_fusion_frames <= 0 else None
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    try:
        run(args)
    except InterruptedRun as exc:
        resume_command = f'lecture2pdf "{args.video}" --output "{args.output}" --resume'
        p.exit(130, f"INTERRUPTED: {exc}\nResume with: {resume_command}\n")
    except (OSError, ValueError, RuntimeError) as exc:
        p.exit(1, f"ERROR: {exc}\n")

# v0.2 orchestration.  It is defined after the v0.1 implementation to keep
# the original small helpers import-compatible for downstream callers.
import shutil
import time
from .acceleration.device import CudaUnavailableError, select_device
from .acceleration.cuda_backend import CUDABackend
from .runtime import CacheLayout, CheckpointStore, ProgressTracker, atomic_write_json, fingerprint
from .video.metadata import probe_video
from .video.decoder import choose_decoder
from .reconstruct import reconstruct_with_metadata
from .ocr import extract_cached
from .cleanup import cleanup_all, cleanup_generated


def _preset(name):
    return {
        "fast": {"sample_fps": 1.0, "change_threshold": 0.25, "fusion_threshold": 0.30, "stable_samples": 1, "max_fusion_frames": 9},
        "balanced": {"sample_fps": 1.5, "change_threshold": 0.20, "fusion_threshold": 0.26, "stable_samples": 2, "max_fusion_frames": 17},
        "maximum": {"sample_fps": 2.0, "change_threshold": 0.19, "fusion_threshold": 0.24, "stable_samples": 2, "max_fusion_frames": 31},
    }[name]


def _v2_run(args):
    src = Path(args.video)
    if not src.is_file():
        raise FileNotFoundError(src)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    device = select_device(args.compute)
    metadata = probe_video(src)
    decoder = choose_decoder(src, args.decoder, "cuda" if device.is_cuda else "cpu")
    defaults = _preset(args.quality)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    config = {k: getattr(args, k) for k in ("quality", "mode", "compute", "decoder", "ocr_engine", "reconstruction", "sample_fps", "change_threshold", "fusion_threshold", "stable_samples", "max_fusion_frames", "roi", "ocr_lang", "no_ocr")}
    job_id = fingerprint(src, config)[:24]
    cache_root = Path(args.cache_dir) if args.cache_dir else out.parent / ".lecture2pdf-cache" / "jobs"
    layout = CacheLayout(cache_root / job_id)
    if args.clear_cache and layout.root.exists():
        shutil.rmtree(layout.root)
    store = CheckpointStore(layout, job_id, job_id)
    progress = ProgressTracker(metadata.duration, verbose=args.verbose, quiet=args.quiet, log_file=args.log_file, device=device)
    backend = CUDABackend(device.info) if device.is_cuda else None
    detector = SlideDetector(change_threshold=args.change_threshold, min_stable=args.stable_samples, backend=backend)
    slides = []
    checkpoint = store.get("detection", {}) if args.resume else {}
    if checkpoint.get("slides"):
        for item in checkpoint["slides"]:
            frame = cv2.imread(str(layout.slides / item["filename"]))
            if frame is not None:
                slides.append({"time": item["time"], "frame": frame, "frames": [], "decision": item.get("decision", {})})
        if slides:
            detector.current = slides[-1]["frame"].copy()
            detector.seen = [s["frame"].copy() for s in slides]
    start_time = float(checkpoint.get("last_sample_time", 0.0)) if args.resume else 0.0
    window = []
    processed = int(checkpoint.get("sample_count", 0)) if args.resume else 0
    duplicates = int(checkpoint.get("duplicates", 0)) if args.resume else 0
    profile_times = {}
    try:
        detect_started = time.perf_counter()
        device_name = device.info.name or device.info.provider or device.device
        progress.update(f"Detecting slides ({device.device.upper()} / {device_name}; decoder={decoder.selected})", start_time, processed, len(slides), duplicates, True)
        for sample in sample_video(src, fps=args.sample_fps, roi=args.roi, decoder=decoder.selected, start_time=start_time):
            processed += 1
            decision = detector.push(sample)
            if decision is not None:
                if slides:
                    slides[-1]["frames"] = window[:]
                stamp, frame = decision
                filename = f"slide_{len(slides)+1:04d}.png"
                if not cv2.imwrite(str(layout.slides / filename), frame):
                    raise OSError(f"Failed to write {layout.slides / filename}")
                slides.append({"time": stamp, "frame": frame, "frames": [], "decision": detector.last_decision or {}})
                window = []
                progress.slide_count = len(slides)
                LOG.info("Unique slide %d at %.1fs", len(slides), stamp)
            elif detector.last_decision and detector.last_decision.get("kind") == "duplicate":
                duplicates += 1
            if slides and difference(sample.frame, slides[-1]["frame"]) < args.fusion_threshold:
                if len(window) < args.max_fusion_frames:
                    window.append(sample.frame.copy())
                else:
                    window.pop(0)
                    window.append(sample.frame.copy())
            progress.update("Detecting slides", sample.time, processed, len(slides), duplicates)
            if processed % 10 == 0:
                store.put("detection", {"last_sample_time": sample.time, "sample_count": processed, "duplicates": duplicates,
                                         "slides": [{"time": s["time"], "filename": f"slide_{i+1:04d}.png", "decision": s.get("decision", {})} for i, s in enumerate(slides)]})
        if slides:
            slides[-1]["frames"] = window
        if not slides:
            raise RuntimeError("No decodable frames found")
        store.put("detection", {"last_sample_time": metadata.duration, "sample_count": processed, "duplicates": duplicates,
                                 "slides": [{"time": s["time"], "filename": f"slide_{i+1:04d}.png", "decision": s.get("decision", {})} for i, s in enumerate(slides)]})
        profile_times["detection_seconds"] = time.perf_counter() - detect_started
        output_slides = []
        recon_started = time.perf_counter()
        for i, slide in enumerate(slides):
            recon_path = layout.reconstructed / f"slide_{i+1:04d}.png"
            original_path = layout.slides / f"slide_{i+1:04d}.png"
            if recon_path.is_file():
                clean = cv2.imread(str(recon_path))
            else:
                if args.reconstruction == "off" or len(slide["frames"]) < 3:
                    clean = slide["frame"]
                    confidence = 1.0
                else:
                    result = reconstruct_with_metadata(slide["frames"], max_frames=args.max_fusion_frames)
                    clean, confidence = result["frame"], result["confidence"]
                if not cv2.imwrite(str(recon_path), clean):
                    raise OSError(f"Failed to write {recon_path}")
                if not original_path.is_file():
                    cv2.imwrite(str(original_path), slide["frame"])
            if args.no_ocr or args.mode == "slides":
                text = ""
            else:
                ocr_file = layout.ocr / f"slide_{i+1:04d}.json"
                if ocr_file.is_file():
                    data = json.loads(ocr_file.read_text(encoding="utf-8"))
                    text = data.get("text", "")
                else:
                    result = extract_cached(clean, layout.ocr, language=args.ocr_lang, engine=args.ocr_engine,
                                            compute="cuda" if device.is_cuda else "cpu")
                    text = result.text
            output_slides.append({"time": slide["time"], "frame": clean, "text": text,
                                  "decision": slide.get("decision", {})})
            progress.update("OCR and reconstruction", slide["time"], i + 1, len(slides), duplicates, True)
        profile_times["reconstruction_ocr_seconds"] = time.perf_counter() - recon_started
        build_pdf(output_slides, out, title=args.title)
        profile_times["pdf_seconds"] = time.perf_counter() - recon_started - profile_times["reconstruction_ocr_seconds"]
        if args.profile:
            LOG.info("Profile timings: %s", json.dumps(profile_times, sort_keys=True))
        legacy = out.parent / f"{out.stem}_assets"
        legacy.mkdir(parents=True, exist_ok=True)
        manifest = [{"slide": i + 1, "timestamp_s": s["time"], "image": f"slide_{i+1:04d}.png", "ocr_text": s["text"], "decision": s.get("decision", {})} for i, s in enumerate(output_slides)]
        for i in range(len(output_slides)):
            shutil.copy2(layout.reconstructed / f"slide_{i+1:04d}.png", legacy / f"slide_{i+1:04d}.png")
        atomic_write_json(legacy / "manifest.json", manifest)
        atomic_write_json(layout.root / "manifest.json", manifest)
        store.put("completed", {"output": str(out), "slides": len(output_slides), "completed_at": time.time()})
        LOG.info("Wrote %s (%d slides) using %s / %s", out, len(output_slides), device.device, device.info.name or device.info.provider or "CPU")
        return out
    except KeyboardInterrupt:
        try:
            store.put("detection", {"last_sample_time": getattr(progress, "last_timestamp", 0.0), "sample_count": processed, "duplicates": duplicates,
                                     "slides": [{"time": s["time"], "filename": f"slide_{i+1:04d}.png", "decision": s.get("decision", {})} for i, s in enumerate(slides)]})
            LOG.warning("Checkpoint saved. Resume with: lecture2pdf %s --output %s --resume", src, out)
        except Exception as exc:
            LOG.error("Checkpoint persistence failed: %s", exc)
        raise
    finally:
        progress.close()
        store.close()


def main(argv=None):
    p = argparse.ArgumentParser(prog="lecture2pdf", description="Extract lecture slides into a searchable study PDF")
    p.add_argument("video", nargs="?", help="Input MP4/MKV/MOV/WebM video")
    p.add_argument("-o", "--output", default="notes.pdf")
    p.add_argument("--title", default="Lecture Study Notes")
    p.add_argument("--quality", choices=["fast", "balanced", "maximum"], default="maximum")
    p.add_argument("--mode", choices=["slides", "study"], default="study")
    p.add_argument("--compute", choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument("--decoder", choices=["auto", "opencv", "ffmpeg", "nvdec"], default="auto")
    p.add_argument("--ocr-engine", choices=["auto", "tesseract", "paddle"], default="auto")
    p.add_argument("--reconstruction", choices=["off", "temporal", "temporal-segmentation"], default="temporal")
    p.add_argument("--sample-fps", type=float, default=None)
    p.add_argument("--change-threshold", type=float, default=None)
    p.add_argument("--fusion-threshold", type=float, default=None)
    p.add_argument("--stable-samples", type=int, default=None)
    p.add_argument("--max-fusion-frames", type=int, default=None)
    p.add_argument("--roi", type=parse_roi, help="Crop normalized x,y,w,h e.g. 0.04,0.05,0.92,0.82")
    p.add_argument("--ocr-lang", default="eng")
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--keep-cache", action="store_true", help="Keep the disk cache (kept by default for resumability)")
    p.add_argument("--clear-cache", action="store_true")
    p.add_argument("--log-file")
    p.add_argument("--profile", action="store_true", help="Enable stage timing logs")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--cleanup", action="store_true", help="Remove Lecture2PDF caches and build artifacts under --cleanup-root")
    p.add_argument("--cleanup-all", action="store_true", help="Also remove generated *_assets directories and local logs/temp files")
    p.add_argument("--cleanup-root", type=Path, default=Path.cwd(), help="Root directory to clean (default: current directory)")
    args = p.parse_args(argv)
    if args.cleanup or args.cleanup_all:
        report = cleanup_all(args.cleanup_root) if args.cleanup_all else cleanup_generated(args.cleanup_root)
        print(f"Cleanup removed {report.removed_count} item(s) under {Path(args.cleanup_root).resolve()}")
        return
    if not args.video:
        p.error("video is required unless --cleanup or --cleanup-all is used")
    if args.quiet and args.verbose:
        p.error("--quiet and --verbose cannot be used together")
    for name in ("sample_fps", "stable_samples", "max_fusion_frames"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", filename=args.log_file)
    try:
        _v2_run(args)
    except (OSError, ValueError, RuntimeError, CudaUnavailableError) as exc:
        p.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
