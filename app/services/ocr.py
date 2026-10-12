"""Extraction de texte : texte natif PDF page par page, OCR (PaddleOCR) si besoin."""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings

log = logging.getLogger(__name__)
_engine = None
_lock = threading.Lock()  # PaddleOCR n'est pas thread-safe : on sérialise
LANG_MAP = {"fra": "fr", "eng": "en", "fra+eng": "fr"}
MAX_IMAGE_SIDE = 3000


@dataclass
class Extraction:
    text: str
    pages: int
    ocr_pages: int


def _get_engine():
    global _engine
    if _engine is None:
        from paddleocr import PaddleOCR

        lang = LANG_MAP.get(get_settings().ocr_lang, "fr")
        log.info("Chargement de PaddleOCR (lang=%s)", lang)
        _engine = PaddleOCR(
            lang=lang,
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            enable_mkldnn=False,
        )
    return _engine


def _parse(result) -> str:
    lines: list[str] = []
    for page in result or []:
        if not page:
            continue
        if hasattr(page, "get") and page.get("rec_texts"):  # PaddleOCR 3.x
            lines.extend(page["rec_texts"])
        elif isinstance(page, list):  # ancien format
            for item in page:
                try:
                    lines.append(item[1][0])
                except (IndexError, TypeError):
                    continue
    return "\n".join(lines)


def run_ocr(image) -> str:
    with _lock:
        engine = _get_engine()
        result = engine.predict(image) if hasattr(engine, "predict") else engine.ocr(image)
    return _parse(result)


def _pil_to_array(img):
    import numpy as np

    img = img.convert("RGB")
    if max(img.size) > MAX_IMAGE_SIDE:
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return np.array(img)


def extract_image(path: Path) -> Extraction:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 60_000_000  # protège des « decompression bombs »
    with Image.open(path) as img:
        arr = _pil_to_array(img)
    return Extraction(run_ocr(arr), pages=1, ocr_pages=1)


def extract_pdf(path: Path) -> Extraction:
    import fitz
    from PIL import Image

    s = get_settings()
    parts: list[str] = []
    ocr_pages = 0
    with fitz.open(path) as doc:
        if doc.needs_pass:
            raise ValueError("PDF protégé par mot de passe")
        if len(doc) > s.max_pdf_pages:
            raise ValueError(f"PDF trop long ({len(doc)} pages, max {s.max_pdf_pages})")
        zoom = s.pdf_render_dpi / 72
        for page in doc:
            native = page.get_text().strip()
            if len(native) >= s.min_chars_per_page:
                parts.append(native)
                continue
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            parts.append(run_ocr(_pil_to_array(img)))
            ocr_pages += 1
        pages = len(doc)
    return Extraction("\n".join(parts), pages=pages, ocr_pages=ocr_pages)


def extract(path: Path, file_type: str) -> Extraction:
    return extract_pdf(path) if file_type == "pdf" else extract_image(path)
