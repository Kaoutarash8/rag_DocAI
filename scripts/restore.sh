#!/bin/sh
# Restaure une sauvegarde sur CETTE installation docker compose (écrase la base et les fichiers).
#   ./scripts/restore.sh backups/20261025T020000Z
# À lancer depuis la racine du projet, sur l'hôte. Testez-le régulièrement sur une machine vierge :
# une sauvegarde jamais restaurée n'est pas une sauvegarde.
set -eu
dir="${1:?usage: restore.sh <dossier de sauvegarde>}"
[ -f "$dir/pg.dump" ] && [ -f "$dir/appdata.tar.gz" ] || { echo "Sauvegarde incomplète : $dir" >&2; exit 1; }
(cd "$dir" && sha256sum -c SHA256SUMS)
printf "Écraser la base et les fichiers actuels avec %s ? [oui/NON] " "$dir"
read -r answer
[ "$answer" = "oui" ] || { echo "Annulé."; exit 1; }

docker compose stop caddy api worker
docker compose up -d db
docker compose exec -T db sh -c 'until pg_isready -U "$POSTGRES_USER" >/dev/null 2>&1; do sleep 1; done'
docker compose exec -T db sh -c 'dropdb -U "$POSTGRES_USER" --if-exists "$POSTGRES_DB" && createdb -U "$POSTGRES_USER" "$POSTGRES_DB"'
docker compose exec -T db sh -c 'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --no-owner' < "$dir/pg.dump"
docker compose run --rm --no-deps -v "$PWD/$dir:/restore:ro" --entrypoint sh api \
  -c 'tar -xzf /restore/appdata.tar.gz -C /data'
if [ -f "$dir/chroma.tar.gz" ]; then
  docker compose stop chroma
  docker compose run --rm --no-deps -v "$PWD/$dir:/restore:ro" --entrypoint sh chroma \
    -c 'rm -rf /chroma/chroma/* && tar -xzf /restore/chroma.tar.gz -C /chroma/chroma' || \
    echo "Restauration des vecteurs impossible : utilisez « python -m scripts.reindex »."
else
  echo "Pas de vecteurs dans cette sauvegarde : après le redémarrage, lancez"
  echo "  docker compose run --rm api python -m scripts.reindex"
fi
docker compose up -d
echo "Restauration terminée."
