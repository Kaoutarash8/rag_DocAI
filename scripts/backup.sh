#!/bin/sh
# Sauvegarde périodique : PostgreSQL (pg_dump), fichiers (uploads + texte extrait) et, en option, vecteurs.
# Tourne dans le service `backup` du docker-compose ; `backup.sh once` fait une seule sauvegarde.
# Les vecteurs sont reconstructibles (`python -m scripts.reindex`) : par défaut on ne les sauvegarde pas,
# car une copie à chaud de ChromaDB peut être incohérente. BACKUP_INCLUDE_VECTORS=true pour les inclure.
set -eu

BACKUP_DIR="${BACKUP_DIR:-/backups}"
KEEP_DAYS="${BACKUP_KEEP_DAYS:-7}"
INTERVAL_H="${BACKUP_INTERVAL_HOURS:-24}"

# tar renvoie 1 si un fichier a changé pendant la lecture (ingestion en cours) : acceptable, pas fatal.
tar_ok() { "$@" || [ $? -eq 1 ]; }

backup_once() {
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  tmp="$BACKUP_DIR/.$ts.partial"
  mkdir -p "$tmp"
  echo "[backup] $ts : PostgreSQL"
  pg_dump -h db -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc -f "$tmp/pg.dump"
  echo "[backup] $ts : fichiers (uploads, texte extrait)"
  tar_ok tar -C /data -czf "$tmp/appdata.tar.gz" --exclude=./cache --exclude=./vectorstore .
  if [ "${BACKUP_INCLUDE_VECTORS:-false}" = "true" ] && [ -d /chroma ]; then
    echo "[backup] $ts : vecteurs"
    tar_ok tar -C /chroma -czf "$tmp/chroma.tar.gz" .
  fi
  (cd "$tmp" && sha256sum ./* > SHA256SUMS)
  mv "$tmp" "$BACKUP_DIR/$ts"   # le dossier n'apparaît sous son vrai nom que lorsqu'il est complet
  find "$BACKUP_DIR" -mindepth 1 -maxdepth 1 -type d -name '20*' -mtime +"$KEEP_DAYS" -exec rm -rf {} +
  echo "[backup] $ts : terminé"
}

if [ "${1:-}" = "once" ]; then
  backup_once
  exit 0
fi
while true; do
  backup_once || echo "[backup] ÉCHEC de la sauvegarde" >&2
  sleep $((INTERVAL_H * 3600))
done
