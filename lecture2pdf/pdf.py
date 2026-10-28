from io import BytesIO
from xml.sax.saxutils import escape
import cv2
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, KeepTogether
from reportlab.lib.units import inch


def build_pdf(slides, destination, title="Lecture Notes", author="Lecture2PDF", subject="Searchable lecture slides"):
    doc = SimpleDocTemplate(str(destination), pagesize=(595.28, 841.89),
                            leftMargin=42, rightMargin=42, topMargin=48, bottomMargin=48,
                            title=title, author=author, subject=subject)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SlideTitleCustom", parent=styles["Heading2"], fontSize=15, spaceAfter=8, textColor=colors.HexColor("#18263a")))
    styles.add(ParagraphStyle(name="OCRCustom", parent=styles["BodyText"], fontSize=10, leading=15, spaceBefore=9, wordWrap="CJK"))
    story = [Paragraph(escape(title), styles["Title"]), Spacer(1, 12),
             Paragraph(f"{len(slides)} unique slides • Source-derived notes", styles["Normal"]), PageBreak()]
    usable_w = doc.width
    usable_h = 490
    for i, slide in enumerate(slides):
        frame = slide["frame"]
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError(f"Cannot encode slide {i+1}")
        bio = BytesIO(encoded.tobytes())
        h, w = frame.shape[:2]
        scale = min(usable_w/w, usable_h/h)
        img = Image(bio, width=w*scale, height=h*scale)
        mins, secs = divmod(int(slide["time"]), 60)
        heading = Paragraph(f"Slide {i+1}  <font color='#667085' size='9'>[{mins:02}:{secs:02}]</font>", styles["SlideTitleCustom"])
        text = slide.get("text", "").strip()
        if not text:
            text = "No reliably recognized text. Refer to the original slide image."
        text = escape(text)
        # Split long OCR output across pages, without clipping the slide image.
        story.extend([heading, img, Spacer(1, 12), Paragraph("<b>Extracted slide text</b>", styles["Normal"]),
                      Paragraph(text, styles["OCRCustom"])])
        if i != len(slides)-1:
            story.append(PageBreak())
    doc.build(story)


# v0.2 implementation kept below the original function so older imports and
# generated manifests remain compatible while gaining richer metadata/TOC.
def build_pdf(slides, destination, title="Lecture Notes", author="Lecture2PDF", subject="Searchable lecture slides"):
    doc = SimpleDocTemplate(str(destination), pagesize=(595.28, 841.89),
                            leftMargin=42, rightMargin=42, topMargin=48, bottomMargin=48,
                            title=title, author=author, subject=subject)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SlideTitleV2", parent=styles["Heading2"], fontSize=15, spaceAfter=8, textColor=colors.HexColor("#18263a")))
    styles.add(ParagraphStyle(name="OCRV2", parent=styles["BodyText"], fontSize=10, leading=15, spaceBefore=9, wordWrap="CJK"))
    story = [Paragraph(escape(title), styles["Title"]), Spacer(1, 12),
             Paragraph(f"{len(slides)} unique slides - source-derived notes", styles["Normal"]),
             Spacer(1, 18), Paragraph("Table of contents", styles["Heading2"])]
    for i, slide in enumerate(slides):
        mins, secs = divmod(int(slide["time"]), 60)
        story.append(Paragraph(f"Slide {i+1} - [{mins:02}:{secs:02}]", styles["Normal"]))
    story.append(PageBreak())
    usable_w, usable_h = doc.width, 490
    for i, slide in enumerate(slides):
        frame = slide["frame"]
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError(f"Cannot encode slide {i+1}")
        h, w = frame.shape[:2]
        scale = min(usable_w / w, usable_h / h)
        mins, secs = divmod(int(slide["time"]), 60)
        heading = Paragraph(f"<a name='slide_{i+1}'/>Slide {i+1}  <font color='#667085' size='9'>[{mins:02}:{secs:02}]</font>", styles["SlideTitleV2"])
        text = escape(slide.get("text", "").strip() or "No reliably recognized text. Refer to the original slide image.")
        story.extend([heading, Image(BytesIO(encoded.tobytes()), width=w * scale, height=h * scale), Spacer(1, 12),
                      Paragraph("<b>Extracted slide text</b>", styles["Normal"]), Paragraph(text, styles["OCRV2"])])
        if i != len(slides) - 1:
            story.append(PageBreak())
    doc.build(story)
