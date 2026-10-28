"""Point d'entrée unique : choisit l'extracteur selon le type de fichier."""

from collections.abc import Iterator
from pathlib import Path

from app.services.content import PageContent


def iter_pages(path: Path, file_type: str) -> tuple[int, Iterator[PageContent]]:
    """(nombre d'unités estimé, générateur d'unités). Une unité = page, diapo, lot de lignes..."""
    if file_type == "pdf":
        from app.services import ocr
        return ocr.iter_pdf(path)
    if file_type == "image":
        from app.services import ocr
        return ocr.iter_image(path)
    from app.services import office
    if file_type == "docx":
        return office.iter_docx(path)
    if file_type == "pptx":
        return office.iter_pptx(path)
    if file_type == "xlsx":
        return office.iter_xlsx(path)
    if file_type in ("txt", "md"):
        return office.iter_text(path, markdown=file_type == "md")
    raise ValueError(f"Type de fichier non géré : {file_type}")
