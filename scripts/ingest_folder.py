"""Importe un dossier de documents (PDF, images, DOCX, PPTX, XLSX, TXT, MD) pour un utilisateur :

    python -m scripts.ingest_folder data/raw --owner prenom@exemple.com
"""
import argparse
import hashlib
import shutil
import uuid
from pathlib import Path

from app.core.config import get_settings
from app.database import SessionLocal
from app.models.user import User
from app.services import processing
from app.services.files import ALLOWED_EXT, sanitize_filename, sniff_file_type
from app.services.ingestion import create_document


def main(folder: str, owner_email: str) -> None:
    s = get_settings()
    processing.start()
    count = 0
    with SessionLocal() as db:
        owner = db.query(User).filter(User.email == owner_email.strip().lower()).first()
        if not owner:
            raise SystemExit(f"Utilisateur introuvable : {owner_email}")
        for f in sorted(Path(folder).rglob("*")):
            if not f.is_file() or f.suffix.lower() not in ALLOWED_EXT:
                continue
            data = f.read_bytes()
            ftype = sniff_file_type(data[:16], f.suffix, f)
            if not ftype:
                print(f"ignoré (contenu invalide) : {f}")
                continue
            tmp = s.upload_dir / f".tmp-{uuid.uuid4().hex}"
            shutil.copyfile(f, tmp)
            doc, created = create_document(db, tmp, sanitize_filename(f.name), ftype,
                                           hashlib.sha256(data).hexdigest(), len(data), owner.id)
            if created:
                processing.submit(doc.id)
                count += 1
            print(("importé" if created else "déjà présent"), f)
    print(f"{count} document(s) en cours de traitement...")
    processing.wait_idle()  # thread : attend la fin ; celery : les workers continuent seuls


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="data/raw")
    ap.add_argument("--owner", required=True, help="e-mail du compte propriétaire des documents importés")
    a = ap.parse_args()
    main(a.folder, a.owner)
