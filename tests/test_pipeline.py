from pathlib import Path
import cv2
import numpy as np
from lecture2pdf.detect import difference, is_duplicate, SlideDetector
from lecture2pdf.reconstruct import reconstruct
from lecture2pdf.video import Sample
from lecture2pdf.pdf import build_pdf
from lecture2pdf.cli import parse_roi

def slide(text):
    a = np.full((360, 640, 3), 25, dtype=np.uint8)
    cv2.putText(a, text, (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4)
    return a

def test_similarity():
    a, b = slide("INTRO"), slide("END")
    assert difference(a,a) < 0.001
    assert is_duplicate(a,a)
    assert difference(a,b) > difference(a,a)

def test_detector_stable_transition():
    detector = SlideDetector(change_threshold=.005, stable_threshold=.03, min_stable=2)
    a, b = slide("AAAA"), 255 - slide("BBBB")
    results = [detector.push(Sample(i, f)) for i,f in enumerate([a,a,a,b,b,b])]
    assert results[0] is not None
    assert sum(r is not None for r in results) == 2

def test_temporal_median_recovers_moving_occluder():
    base = slide("HELLO")
    frames = []
    for x in [0,180,360,480,0,180,360]:
        a = base.copy()
        cv2.rectangle(a, (x, 0), (x+70, 350), (50,50,200), -1)
        frames.append(a)
    result = reconstruct(frames)
    assert np.mean(cv2.absdiff(result,base)) < np.mean(cv2.absdiff(frames[1],base))

def test_pdf(tmp_path):
    dest = tmp_path / "notes.pdf"
    build_pdf([{"frame":slide("HELLO"),"time":13.5,"text":"Hello from OCR"}], dest)
    assert dest.read_bytes().startswith(b"%PDF")
    assert dest.stat().st_size > 1000

def test_roi():
    assert parse_roi("0.1,0.2,0.8,0.7") == (.1,.2,.8,.7)
