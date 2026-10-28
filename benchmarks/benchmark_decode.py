"""Measure sequential sampling throughput without making performance claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from lecture2pdf.acceleration.device import detect_cuda
from lecture2pdf.video import sample_video, probe_video


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--decoder", choices=["auto", "opencv", "ffmpeg", "nvdec"], default="auto")
    args = parser.parse_args(argv)
    meta = probe_video(args.video)
    start = time.perf_counter()
    count = 0
    for _ in sample_video(args.video, fps=args.sample_fps, decoder=args.decoder):
        count += 1
    elapsed = time.perf_counter() - start
    result = {"video": str(args.video), "decoder": args.decoder, "duration_s": meta.duration,
              "sampled_frames": count, "wall_seconds": elapsed,
              "sampled_fps": count / elapsed if elapsed else 0.0,
              "decode_video_seconds_per_wall_second": meta.duration / elapsed if elapsed else 0.0,
              "cuda_probe": detect_cuda().available}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
