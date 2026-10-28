import cv2
import pytesseract
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path

@dataclass
class OCRWord:
    text: str
    confidence: float
    box: tuple[int, int, int, int]

@dataclass
class OCRResult:
    text: str
    words: list[OCRWord]
    engine: str

    def to_dict(self):
        return {"text": self.text, "engine": self.engine, "words": [asdict(w) for w in self.words]}

def select_engine(requested="auto", compute="cpu"):
    requested = requested.lower()
    if requested not in {"auto", "tesseract", "paddle"}:
        raise ValueError("ocr engine must be auto, tesseract, or paddle")
    if requested == "paddle" or (requested == "auto" and compute == "gpu"):
        try:
            import paddleocr  # noqa: F401
            return "paddle"
        except Exception as exc:
            if requested == "paddle":
                raise RuntimeError("PaddleOCR was requested but is not installed or failed to import") from exc
    return "tesseract"

def _preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    enlarged = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)).apply(enlarged)
    return enlarged

def extract_result(frame, language="eng", engine="auto", compute="cpu"):
    selected = select_engine(engine, compute)
    if selected == "paddle":
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
            result = ocr.predict(frame)
            words = []
            for item in result:
                data = getattr(item, "json", lambda: item)()
                if isinstance(data, str):
                    data = json.loads(data)
                payload = data.get("res", data) if isinstance(data, dict) else {}
                texts = payload.get("rec_texts", [])
                scores = payload.get("rec_scores", [])
                boxes = payload.get("rec_boxes", [])
                for text, score, box in zip(texts, scores, boxes):
                    if str(text).strip():
                        x0, y0, x1, y1 = [int(v) for v in box[:4]]
                        words.append(OCRWord(str(text).strip(), float(score), (x0, y0, x1-x0, y1-y0)))
            return OCRResult(" ".join(w.text for w in words), words, "paddle")
        except Exception as exc:
            if engine == "paddle":
                raise RuntimeError(f"PaddleOCR failed: {exc}") from exc
            selected = "tesseract"
    enlarged = _preprocess(frame)
    try:
        data = pytesseract.image_to_data(enlarged, lang=language, config="--psm 11", output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError("Tesseract binary missing. Install Tesseract OCR or use --no-ocr.") from exc
    words = []
    for word, confidence, x, y, w, h in zip(data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"]):
        try:
            if word.strip() and float(confidence) >= 35:
                words.append(OCRWord(word.strip(), float(confidence), (int(x), int(y), int(w), int(h))))
        except (TypeError, ValueError):
            pass
    return OCRResult(" ".join(w.text for w in words), words, "tesseract")

def extract(frame, language="eng", engine="auto", compute="cpu"):
    return extract_result(frame, language=language, engine=engine, compute=compute).text

def cache_key(frame, language="eng", engine="auto"):
    digest = hashlib.sha256(frame.tobytes()).hexdigest()
    return f"{digest}-{language}-{engine}"

def extract_cached(frame, cache_dir, language="eng", engine="auto", compute="cpu"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key(frame, language, engine)}.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return OCRResult(data["text"], [OCRWord(w["text"], w["confidence"], tuple(w["box"])) for w in data["words"]], data["engine"])
    result = extract_result(frame, language=language, engine=engine, compute=compute)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")
    temp.replace(path)
    return result
