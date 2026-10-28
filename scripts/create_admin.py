"""Crée (ou promeut) un administrateur et, avec --adopt-orphans, lui attribue les documents sans propriétaire.

    docker compose exec api python -m scripts.create_admin --email admin@exemple.com --adopt-orphans
(le mot de passe est demandé sans écho, ou fourni par la variable ADMIN_PASSWORD)
"""
import argparse
import getpass
import os
import sys

from app.api.schemas import normalize_email
from app.core.auth import hash_password
from app.core.config import get_settings
from app.database import session_scope
from app.models.document import Document
from app.models.user import User, UserRole


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--adopt-orphans", action="store_true", help="attribuer les documents sans propriétaire")
    args = ap.parse_args()
    email = normalize_email(args.email)
    password = os.environ.get("ADMIN_PASSWORD") or getpass.getpass("Mot de passe : ")
    if len(password) < get_settings().password_min_length:
        sys.exit(f"Mot de passe trop court (minimum {get_settings().password_min_length} caractères)")
    with session_scope() as db:
        user = db.query(User).filter(User.email == email).first()
        if user:
            user.role, user.is_active, user.password_hash = UserRole.ADMIN, True, hash_password(password)
            print(f"Compte existant promu administrateur : {email}")
        else:
            user = User(email=email, password_hash=hash_password(password), role=UserRole.ADMIN)
            db.add(user)
            db.flush()
            print(f"Administrateur créé : {email}")
        if args.adopt_orphans:
            n = db.query(Document).filter(Document.owner_id.is_(None)).update({"owner_id": user.id})
            print(f"{n} document(s) sans propriétaire attribué(s) à {email}")


if __name__ == "__main__":
    main()
