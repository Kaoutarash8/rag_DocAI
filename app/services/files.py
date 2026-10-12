"""Validation et nettoyage des fichiers importés."""

import re
from pathlib import Path

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
ALLOWED_EXT = IMAGE_EXT | {".pdf"}


def sanitize_filename(name: str | None) -> str:
    base = Path((name or "document").replace("\\", "/")).name
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip().strip(".")
    if len(base) > 200:
        stem, ext = Path(base).stem[:190], Path(base).suffix[:10]
        base = stem + ext
    return base or "document"


def sniff_file_type(head: bytes, ext: str) -> str | None:
    """Vérifie les octets magiques ET la cohérence avec l'extension."""
    ext = ext.lower()
    if head.startswith(b"%PDF-"):
        return "pdf" if ext == ".pdf" else None
    if ext in IMAGE_EXT:
        is_image = (
            head.startswith(b"\x89PNG\r\n\x1a\n")
            or head.startswith(b"\xff\xd8\xff")
            or (head[:4] == b"RIFF" and head[8:12] == b"WEBP")
            or head.startswith(b"BM")
            or head[:4] in (b"II*\x00", b"MM\x00*")
        )
        return "image" if is_image else None
    return None
