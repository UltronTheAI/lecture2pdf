"""Measure bounded temporal reconstruction; reports measured results only."""
from __future__ import annotations

import argparse
import json
import time
import cv2
import numpy as np
from lecture2pdf.reconstruct import reconstruct_with_metadata


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args(argv)
    base = np.full((360, 640, 3), 30, np.uint8)
    cv2.putText(base, "Reconstruction", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    frames = []
    for x in range(args.frames):
        frame = base.copy()
        cv2.rectangle(frame, (x * 20 % 600, 0), (x * 20 % 600 + 30, 359), (40, 40, 180), -1)
        frames.append(frame)
    start = time.perf_counter()
    confidences = []
    for _ in range(args.iterations):
        confidences.append(reconstruct_with_metadata(frames, max_frames=args.frames)["confidence"])
    elapsed = time.perf_counter() - start
    print(json.dumps({"frames_per_reconstruction": args.frames, "iterations": args.iterations,
                      "wall_seconds": elapsed, "reconstructions_per_second": args.iterations / elapsed if elapsed else 0.0,
                      "mean_confidence": float(np.mean(confidences))}, indent=2))


if __name__ == "__main__":
    main()
