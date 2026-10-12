# Assistant documentaire IA (RAG) — v2

FastAPI + PostgreSQL + ChromaDB + PaddleOCR + LLM (Groq / Ollama / OpenAI).

## Démarrage (Docker)
```bash
cp .env.example .env      # puis éditer : API_KEY, POSTGRES_PASSWORD, GROQ_API_KEY
docker compose up -d --build
curl localhost:8000/ready
```
Interface : http://localhost:8000 — Swagger (hors production) : /docs

## Migrer vos anciens documents
```bash
docker compose exec api python -m scripts.ingest_folder /chemin/vers/dossier
```
(L'ancien index Chroma n'est pas réutilisable : similarité L2 -> cosinus, nouvelle collection.)

## API (en-tête `X-API-Key` si `API_KEY` est défini)
| Route | Rôle |
|---|---|
| `POST /documents/upload` | Import (202 ; 200 si contenu déjà présent) |
| `GET /documents?limit&offset&status` | Liste paginée (`X-Total-Count`) |
| `GET /documents/{id}` · `/text` | Statut · texte extrait |
| `POST /documents/{id}/reprocess` | Relancer un traitement |
| `DELETE /documents/{id}` | Supprime fichier, texte, vecteurs, ligne SQL |
| `POST /ask` | `{question, top_k, document_ids?}` -> réponse + sources citées |
| `GET /health` · `/ready` | Liveness · readiness (DB + vectorstore) |

## Tests
```bash
pip install -r requirements.txt -r requirements-dev.txt && pytest
```

## Limites connues / pour passer à l'échelle
- 1 worker uvicorn : ChromaDB local et la file de traitement sont mono-process.
  Pour plusieurs instances : pgvector ou Chroma serveur + file Celery/RQ.
- Limiteur de débit en mémoire (par process) : utiliser Redis ou le reverse proxy en multi-instances.
- Mettre un reverse proxy HTTPS (Caddy/Nginx/Traefik) devant l'API.
- Sauvegardes : volumes `pgdata` et `appdata`.
