import cv2
import numpy as np
from .detect import difference

def _sharpness(frame):
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _bounded_subset(frames, max_frames):
    """Keep at most ``max_frames`` frames without retaining an unbounded stream."""
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    subset = []
    for index, frame in enumerate(frames):
        if frame is None or not hasattr(frame, "size") or frame.size == 0:
            continue
        if len(subset) < max_frames:
            subset.append(frame)
        else:
            # Deterministic reservoir replacement keeps long segments
            # represented across their whole duration without random tests.
            slot = (index * 2654435761) % (index + 1)
            if slot < max_frames:
                subset[slot] = frame
    if not subset:
        raise ValueError("No usable frames for reconstruction")
    return subset


def _align_with_mask(reference, frame):
    """Align a frame and return a validity mask for pixels observed after warp."""
    target_shape = reference.shape[:2]
    if frame.shape[:2] != target_shape:
        frame = cv2.resize(frame, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
    a = reference if reference.ndim == 2 else cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    b = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)
    valid = np.full(target_shape, 255, dtype=np.uint8)
    try:
        _, warp = cv2.findTransformECC(
            a, b, warp, cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 1e-4),
        )
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        aligned = cv2.warpAffine(frame, warp, (target_shape[1], target_shape[0]),
                                  flags=flags, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
        valid = cv2.warpAffine(valid, warp, (target_shape[1], target_shape[0]),
                               flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return aligned, valid > 0, True
    except cv2.error:
        return frame, valid > 0, False

def align(reference, frame):
    """Translation-only ECC alignment; safely fall back on failure."""
    return _align_with_mask(reference, frame)[0]

def reconstruct(frames, max_frames=31):
    """Fuse a bounded set of frames using only observed pixel values.

    The temporal median is used only to choose the closest *observed* pixel;
    median pixel values are never written to the result. Frames must be from
    the same stable slide; use detection before fusion.
    """
    subset = _bounded_subset(frames, max_frames)
    # Blurry frames reduce the reliability of a temporal median and often
    # contain transition motion. Keep at least one frame if all are blurry.
    scores = [_sharpness(f) for f in subset]
    if len(subset) >= 5 and max(scores) > 1e-6:
        cutoff = max(1e-6, max(scores) * 0.12)
        filtered = [f for f, score in zip(subset, scores) if score >= cutoff]
        if filtered:
            subset = filtered
    scores = [_sharpness(f) for f in subset]
    ref = subset[int(np.argmax(scores))]
    aligned = []
    masks = []
    for frame in subset:
        image, mask, _ = _align_with_mask(ref, frame)
        aligned.append(image)
        masks.append(mask)
    stack = np.stack(aligned)
    valid = np.stack(masks)
    # The median is a selection reference only. Taking argmin against it
    # guarantees every output pixel comes from an actual input frame.
    center = np.median(stack.astype(np.float32), axis=0)
    if stack.ndim == 4:
        distances = np.mean(np.abs(stack.astype(np.float32) - center), axis=3)
    else:
        distances = np.abs(stack.astype(np.float32) - center)
    distances[~valid] = np.inf
    chosen = np.argmin(distances, axis=0)
    rows, cols = np.indices(chosen.shape)
    return stack[chosen, rows, cols].astype(np.uint8)

def reconstruct_with_metadata(frames, max_frames=31):
    """Return the conservative reconstruction and measurable confidence.

    The output is a robust statistic of observed, aligned pixels only. It does
    not infer permanently occluded content. ``representative`` is retained so
    callers can preserve the original evidence next to the fused image.
    """
    subset = _bounded_subset(frames, max_frames)
    representative = np.asarray(subset[len(subset) // 2]).copy()
    fused = reconstruct(subset, max_frames=max_frames)
    residuals = [difference(f, fused) for f in subset]
    confidence = float(max(0.0, min(1.0, 1.0 - np.median(residuals) * 2.0)))
    # High residuals identify regions that remain uncertain; they are reported,
    # not inpainted. The representative frame remains available as evidence.
    gray = [f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in subset]
    uncertainty = float(np.mean(np.std(np.stack(gray).astype(np.float32), axis=0) > 30.0))
    return {"frame": fused, "representative": representative, "confidence": confidence,
            "observed_frames": len(subset), "fabricated": False,
            "uncertain_fraction": uncertainty, "residuals": residuals}

def same_slide_frames(frames, reference, threshold=0.24):
    return [f for f in frames if difference(f, reference) < threshold]
