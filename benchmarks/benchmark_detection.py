"""Measure detector throughput on deterministic synthetic frames."""
from __future__ import annotations

import argparse
import json
import time
import cv2
import numpy as np
from lecture2pdf.detect import SlideDetector
from lecture2pdf.video import Sample


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=240)
    args = parser.parse_args(argv)
    base = np.full((360, 640, 3), 30, np.uint8)
    cv2.putText(base, "Lecture2PDF benchmark", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    detector = SlideDetector()
    start = time.perf_counter()
    slides = 0
    for i in range(args.frames):
        frame = base if (i // 60) % 2 == 0 else 255 - base
        if detector.push(Sample(i / 2.0, frame)) is not None:
            slides += 1
    elapsed = time.perf_counter() - start
    print(json.dumps({"frames": args.frames, "wall_seconds": elapsed,
                      "frames_per_second": args.frames / elapsed if elapsed else 0.0, "slides": slides}, indent=2))


if __name__ == "__main__":
    main()
