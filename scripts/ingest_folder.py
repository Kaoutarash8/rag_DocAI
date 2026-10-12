"""Importe un dossier de PDF/images : python -m scripts.ingest_folder data/raw"""
import hashlib
import shutil
import sys
import uuid
from pathlib import Path

from app.core.config import get_settings
from app.database import SessionLocal
from app.services import processing
from app.services.files import ALLOWED_EXT, sanitize_filename, sniff_file_type
from app.services.ingestion import create_document


def main(folder: str) -> None:
    s = get_settings()
    processing.start()
    count = 0
    with SessionLocal() as db:
        for f in sorted(Path(folder).rglob("*")):
            if not f.is_file() or f.suffix.lower() not in ALLOWED_EXT:
                continue
            data = f.read_bytes()
            ftype = sniff_file_type(data[:16], f.suffix)
            if not ftype:
                print(f"ignoré (contenu invalide) : {f}")
                continue
            tmp = s.upload_dir / f".tmp-{uuid.uuid4().hex}"
            shutil.copyfile(f, tmp)
            doc, created = create_document(db, tmp, sanitize_filename(f.name), ftype,
                                           hashlib.sha256(data).hexdigest(), len(data))
            if created:
                processing.submit(doc.id)
                count += 1
            print(("importé" if created else "déjà présent"), f)
    print(f"{count} document(s) en cours de traitement...")
    processing._executor.shutdown(wait=True)  # attend la fin des traitements


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
