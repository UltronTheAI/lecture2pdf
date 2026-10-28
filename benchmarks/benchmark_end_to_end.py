"""Run a measured end-to-end conversion for CPU/CUDA comparison.

This script invokes the installed CLI with identical flags. It never invents
CUDA numbers: an unavailable CUDA run is reported as unavailable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from lecture2pdf.acceleration.device import detect_cuda


def run(video, output, compute):
    command = [sys.executable, "-m", "lecture2pdf.cli", str(video), "--output", str(output),
               "--quality", "balanced", "--mode", "slides", "--compute", compute,
               "--decoder", "auto", "--reconstruction", "off", "--no-ocr", "--quiet"]
    start = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - start
    return {"compute": compute, "wall_seconds": elapsed, "returncode": completed.returncode,
            "output_size_bytes": output.stat().st_size if output.is_file() else None,
            "stderr_tail": completed.stderr[-500:]}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_outputs"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [run(args.video, args.output_dir / "cpu.pdf", "cpu")]
    if detect_cuda().available:
        results.append(run(args.video, args.output_dir / "cuda.pdf", "gpu"))
    else:
        results.append({"compute": "gpu", "status": "unavailable", "reason": detect_cuda().error})
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
