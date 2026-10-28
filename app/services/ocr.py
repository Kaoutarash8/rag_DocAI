"""Extraction PDF / image : texte natif page par page, titres, tableaux, figures, OCR (PaddleOCR)."""

import hashlib
import logging
import threading
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from app.core.config import get_settings
from app.services import vision
from app.services.content import (HeadingStack, PageContent, Segment, font_profile,
                                  rows_to_markdown)

log = logging.getLogger(__name__)
_engine = None
_lock = threading.Lock()  # PaddleOCR n'est pas thread-safe : on sérialise
LANG_MAP = {"fra": "fr", "eng": "en", "fra+eng": "fr"}
MAX_IMAGE_SIDE = 3000
MIN_FIGURE_SIDE_PT = 80        # figures plus petites que ~3 cm : icônes, puces
MIN_FIGURE_PAGE_RATIO = 0.03   # ... ou < 3 % de la page
MAX_FIGURE_PAGE_RATIO = 0.85   # > 85 % de la page : page scannée, déjà traitée par l'OCR


# ------------------------------------------------------------------ OCR

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


# ------------------------------------------------------------------ titres (fonctions pures)

def _round_size(size: float) -> float:
    return round(size * 2) / 2


def collect_font_sizes(lines: list[dict], counts: Counter) -> None:
    for ln in lines:
        counts[_round_size(ln["size"])] += len(ln["text"])


def is_heading(line: dict, body: float, levels: list[float]) -> int:
    """Niveau de titre (1 = le plus haut) ou 0 si la ligne est du texte courant."""
    text = line["text"].strip()
    if not text or len(text) > 150 or text.replace(" ", "").isdigit():
        return 0
    size = _round_size(line["size"])
    if size in levels:
        return levels.index(size) + 1
    # ligne en gras de la taille du corps, courte et sans ponctuation finale
    if line.get("bold") and size >= body and len(text) <= 80 and not text.endswith((".", ",", ";", ":")):
        return len(levels) + 1
    return 0


def build_segments(lines: list[dict], tables: list[dict], body: float, levels: list[float],
                   stack: HeadingStack) -> list[Segment]:
    """Transforme les lignes d'une page (+ tableaux) en segments typés, rattachés à leur section.

    lines  : {"text", "size", "bold", "y0", "x0"} dans l'ordre de lecture, hors zones de tableau
    tables : {"markdown", "y0"}
    """
    segments: list[Segment] = []
    buf: list[str] = []
    buf_section = stack.path
    pending = sorted(tables, key=lambda t: t["y0"])
    prev_heading_level = 0

    def flush():
        nonlocal buf
        text = "\n".join(buf).strip()
        if text:
            segments.append(Segment(text, "text", buf_section))
        buf = []

    def emit_tables(before_y: float | None):
        nonlocal buf_section
        while pending and (before_y is None or pending[0]["y0"] <= before_y):
            t = pending.pop(0)
            flush()
            segments.append(Segment(t["markdown"], "table", stack.path))
            buf_section = stack.path

    for ln in lines:
        emit_tables(ln["y0"])
        level = is_heading(ln, body, levels)
        if level:
            if level == prev_heading_level and buf and len(buf) == 1:
                # titre sur plusieurs lignes : on fusionne avec la ligne précédente
                buf[0] += " " + ln["text"].strip()
                stack.push(level, buf[0])
                continue
            flush()
            stack.push(level, ln["text"])
            buf_section = stack.path
            buf = [ln["text"].strip()]
            prev_heading_level = level
        else:
            buf.append(ln["text"].strip())
            prev_heading_level = 0
    flush()
    emit_tables(None)
    return segments


# ------------------------------------------------------------------ PDF (PyMuPDF)

def _page_lines(page) -> list[dict]:
    lines = []
    for block in page.get_text("dict", sort=True)["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            spans = [sp for sp in line["spans"] if sp["text"].strip()]
            if not spans:
                continue
            lines.append({
                "text": "".join(sp["text"] for sp in line["spans"]).strip(),
                "size": max(sp["size"] for sp in spans),
                "bold": all((sp["flags"] & 16) or "bold" in sp["font"].lower() for sp in spans),
                "x0": line["bbox"][0], "y0": line["bbox"][1], "bbox": line["bbox"],
            })
    return lines


def _profile_document(doc) -> tuple[float, list[float]]:
    """Taille du corps et des titres, estimées sur un échantillon de pages natives."""
    counts: Counter = Counter()
    n = len(doc)
    step = max(1, n // 40)
    for i in range(0, n, step):
        try:
            collect_font_sizes(_page_lines(doc[i]), counts)
        except Exception:  # noqa: BLE001
            continue
    return font_profile(counts)


def _in_any(bbox, rects, margin: float = 2.0) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(r[0] - margin <= cx <= r[2] + margin and r[1] - margin <= cy <= r[3] + margin
               for r in rects)


def _extract_tables(page) -> list[dict]:
    if not get_settings().extract_tables:
        return []
    out = []
    try:
        for t in page.find_tables().tables:
            md = rows_to_markdown(t.extract())
            if md and md.count("\n") >= 2:  # au moins en-tête + 1 ligne
                out.append({"markdown": md, "y0": t.bbox[1], "bbox": tuple(t.bbox)})
    except Exception as e:  # noqa: BLE001 - find_tables peut échouer sur des PDF atypiques
        log.warning("Détection de tableaux impossible (page %s) : %s", page.number + 1, e)
    return out


def _figure_rects(page, table_boxes: list) -> list:
    """Rectangles d'images raster et de graphiques vectoriels, hors tableaux et pages scannées."""
    import fitz

    page_area = page.rect.width * page.rect.height
    cands = []
    try:
        cands += [fitz.Rect(i["bbox"]) for i in page.get_image_info()]
    except Exception:  # noqa: BLE001
        pass
    if hasattr(page, "cluster_drawings"):
        try:
            cands += list(page.cluster_drawings())
        except Exception:  # noqa: BLE001
            pass
    filtered = []
    for r in cands:
        r = r & page.rect
        area = r.width * r.height
        if (r.is_empty or min(r.width, r.height) < MIN_FIGURE_SIDE_PT
                or not MIN_FIGURE_PAGE_RATIO <= area / page_area <= MAX_FIGURE_PAGE_RATIO):
            continue
        if any((r & fitz.Rect(tb)).get_area() > 0.5 * area for tb in table_boxes):
            continue  # cadre d'un tableau, pas une figure
        filtered.append(r)
    keep: list = []
    for r in sorted(filtered, key=lambda x: -x.get_area()):
        if not any(k.contains(r) for k in keep):  # ignore les rectangles imbriqués
            keep.append(r)
    return keep


def _describe_figures(page, tables, section: str, seen: set, budget: list[int]) -> list[Segment]:
    import fitz

    s = get_settings()
    out = []
    zoom = s.vlm_dpi / 72
    for rect in _figure_rects(page, [t["bbox"] for t in tables]):
        if budget[0] <= 0:
            break
        png = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect).tobytes("png")
        digest = hashlib.md5(png).hexdigest()
        if digest in seen:  # logo/bandeau répété sur chaque page
            continue
        seen.add(digest)
        desc = vision.describe(png, "image/png", section)
        if desc:
            budget[0] -= 1
            out.append(Segment(f"[Figure, page {page.number + 1}] {desc}", "figure", section))
    return out


def iter_pdf(path: Path) -> tuple[int, Iterator[PageContent]]:
    """(nombre de pages, générateur de pages). Le PDF reste ouvert pendant l'itération."""
    import fitz

    s = get_settings()
    doc = fitz.open(path)
    try:
        if doc.needs_pass:
            raise ValueError("PDF protégé par mot de passe")
        total = len(doc)
        if total > s.max_pdf_pages:
            raise ValueError(f"PDF trop long ({total} pages, max {s.max_pdf_pages})")
    except Exception:
        doc.close()
        raise

    def gen() -> Iterator[PageContent]:
        from PIL import Image

        try:
            body, levels = _profile_document(doc)
            stack, seen, budget = HeadingStack(), set(), [s.vlm_max_figures]
            zoom = s.pdf_render_dpi / 72
            for page in doc:
                native = page.get_text().strip()
                if len(native) >= s.min_chars_per_page:
                    tables = _extract_tables(page)
                    boxes = [t["bbox"] for t in tables]
                    lines = [ln for ln in _page_lines(page) if not _in_any(ln["bbox"], boxes)]
                    segs = build_segments(lines, tables, body, levels, stack)
                    if vision.enabled():
                        segs += _describe_figures(page, tables, stack.path, seen, budget)
                    yield PageContent(page.number + 1, segs)
                else:  # page scannée ou quasi vide : OCR
                    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = run_ocr(_pil_to_array(img))
                    yield PageContent(page.number + 1, [Segment(text, "text", stack.path)], ocr=True)
        finally:
            doc.close()

    return total, gen()


def iter_image(path: Path) -> tuple[int, Iterator[PageContent]]:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 60_000_000  # protège des « decompression bombs »

    def gen() -> Iterator[PageContent]:
        with Image.open(path) as img:
            segs = [Segment(run_ocr(_pil_to_array(img)), "text")]
            if vision.enabled():
                import io

                buf = io.BytesIO()
                small = img.convert("RGB")
                small.thumbnail((1600, 1600))
                small.save(buf, "PNG")
                desc = vision.describe(buf.getvalue(), "image/png")
                if desc:
                    segs.append(Segment(f"[Figure] {desc}", "figure"))
        yield PageContent(1, segs, ocr=True)

    return 1, gen()
