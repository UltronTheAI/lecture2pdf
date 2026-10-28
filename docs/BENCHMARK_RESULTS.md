# Lecture2PDF v0.2 measured benchmark results

Date: 2026-10-28  
Host: Windows 11, NVIDIA GeForce RTX 5050 Laptop GPU (CUDA provider operational through the installed optional backend)

## End-to-end CPU versus CUDA

Input: an 18-second H.264 test clip derived from the bundled AV1 lecture video. Both runs used the same balanced preset, `--mode slides`, `--decoder auto`, `--reconstruction off`, `--no-ocr`, and the same output settings. CUDA mode used the real CUDA-backed coarse detector comparison signal; FFmpeg decoding remained software because NVDEC was not requested.

| Metric | CPU | CUDA |
|---|---:|---:|
| End-to-end wall time | 70.70 s | 76.71 s |
| Video duration / wall time | 0.255x | 0.235x |
| Output size | 2,044,740 bytes | 2,044,740 bytes |
| Detected slides | 2 | 2 |
| Peak RAM | Not captured by this run | Not captured by this run |
| Peak VRAM | N/A | Not captured by this run |

Measured CUDA was 8.5% slower for this small AV1 workload. The result is expected: decode and most structural comparison work are CPU-side, while the coarse GPU signal adds transfer and kernel-launch overhead. This is not a GPU speedup claim. Larger batches, GPU-native preprocessing, or model inference are more likely to amortize that overhead.

## Component measurements

These are separate deterministic measurements, not extrapolations:

* Detection: 30 synthetic frames in 1.1951 s, 25.10 frames/s.
* Reconstruction: one five-frame reconstruction in 0.3847 s, 2.599 reconstructions/s, mean confidence 0.8978.
* Sequential decode/sample: 9 samples from the 18-second clip in 80.6020 s, 0.1117 samples/s, 0.2233 video-seconds per wall-second. The slow result reflects AV1 software decode on this environment and is why the decoder subsystem does not assume hardware acceleration is faster.

The benchmark scripts print fresh JSON and do not fabricate unavailable CUDA, RAM, or VRAM values. Run `benchmarks/benchmark_end_to_end.py`, `benchmark_decode.py`, `benchmark_detection.py`, and `benchmark_reconstruction.py` on a representative production video before making deployment claims.
