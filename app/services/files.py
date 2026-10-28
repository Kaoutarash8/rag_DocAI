"""Validation et nettoyage des fichiers importés."""

import re
from pathlib import Path

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
OFFICE_EXT = {".docx": "docx", ".pptx": "pptx", ".xlsx": "xlsx"}
TEXT_EXT = {".txt": "txt", ".md": "md", ".markdown": "md"}
ALLOWED_EXT = IMAGE_EXT | {".pdf"} | set(OFFICE_EXT) | set(TEXT_EXT)
SUPPORTED_LABEL = "PDF, images, DOCX, PPTX, XLSX, TXT, MD"


def sanitize_filename(name: str | None) -> str:
    base = Path((name or "document").replace("\\", "/")).name
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip().strip(".")
    if len(base) > 200:
        stem, ext = Path(base).stem[:190], Path(base).suffix[:10]
        base = stem + ext
    return base or "document"


def sniff_file_type(head: bytes, ext: str, path: Path | None = None) -> str | None:
    """Vérifie les octets magiques ET la cohérence avec l'extension.

    `path` (fichier déjà enregistré) permet les contrôles approfondis : structure de
    l'archive Office, absence d'octets nuls dans un fichier texte.
    """
    ext = ext.lower()
    if ext in OFFICE_EXT:
        if not head.startswith(b"PK\x03\x04"):
            return None
        kind = OFFICE_EXT[ext]
        if path is not None:
            from app.services.office import check_office_zip
            if not check_office_zip(path, kind):
                return None
        return kind
    if ext in TEXT_EXT:
        if path is not None:
            with path.open("rb") as f:
                if b"\x00" in f.read(8192):
                    return None
        elif b"\x00" in head:
            return None
        return TEXT_EXT[ext]
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
