# Assistant documentaire IA (RAG) — v2

FastAPI + PostgreSQL + ChromaDB + PaddleOCR + LLM (Groq / Ollama / OpenAI).

## Démarrage (Docker, production : HTTPS + workers)
```bash
cp .env.example .env      # puis éditer : JWT_SECRET, POSTGRES_PASSWORD, REDIS_PASSWORD, GROQ_API_KEY, DOMAIN
docker compose up -d --build
docker compose exec api python -m scripts.create_admin --email vous@exemple.com --adopt-orphans
curl -k https://localhost/ready
```
Interface : https://DOMAIN (certificat local si `DOMAIN=localhost`, Let's Encrypt sinon). Swagger (hors production) : /docs.
Seuls les ports 80/443 (Caddy) sont publiés ; API, PostgreSQL, Redis et Chroma restent sur le réseau Docker.

Développement local (un seul conteneur, sans Redis/Celery) : `docker compose -f docker-compose.dev.yml up --build` -> http://localhost:8000 (`ENV=development`).

### Mise à niveau depuis la v2.2 (sans comptes)
1. `docker compose up -d --build` : l'API exécute `alembic upgrade head` (0003 : comptes, `owner_id`, partages ; 0004 : versions d'index).
2. Les documents existants n'ont pas de propriétaire : ils ne sont visibles que des admins. `create_admin --adopt-orphans` les attribue.
3. Les collections portent désormais un suffixe de version (`documents_v2`, `docs_e5_fixed_v2`...). La collection `baseline` garde son nom
   (`documents_v2`) ; les collections de comparaison (`docs_*`) doivent être recréées (`python -m scripts.reindex`).
4. La clé d'API (`X-API-Key`) disparaît au profit de l'authentification par compte.

## Migrer vos anciens documents
```bash
docker compose exec api python -m scripts.ingest_folder /chemin/vers/dossier --owner vous@exemple.com
```

## Sécurité et comptes
| Route | Rôle |
|---|---|
| `POST /auth/register` | Inscription (compte standard ; `ALLOW_REGISTRATION=false` la désactive) |
| `POST /auth/login` | `{email, password}` -> `{access_token, expires_in}` (JWT HS256, `Authorization: Bearer ...`) |
| `GET /auth/me` | Compte courant |
| `GET/POST /admin/users` · `PATCH /admin/users/{id}` | Liste, création, rôle, désactivation (admin) |
| `GET /admin/system` | Versions en service : app, index, modèles, LLM, registre des collections (admin) |

- **Mots de passe** : scrypt (N=2^15, r=8, p=3), sel aléatoire, paramètres stockés dans le hachage ; même coût si l'e-mail est inconnu.
- **JWT** : durée `ACCESS_TOKEN_MINUTES` ; le rôle et l'état actif sont relus en base à chaque requête (désactiver un compte coupe l'accès
  immédiatement). Pas de refresh token ni de liste de révocation : les jetons sont courts, `logout` côté client oublie le jeton.
- **Droits sur les documents** : le propriétaire lit, retraite, supprime et partage ; un destinataire de partage lit (liste, texte, résumé,
  `/ask`) ; un admin lit et gère tout via `/documents`. Document inaccessible = **404** (jamais 403). La recherche `/ask` d'un admin reste
  limitée à ses documents sauf `{"scope": "all"}`.
- **Partages** : `POST /documents/{id}/shares {email}` · `GET` · `DELETE /documents/{id}/shares/{user_id}`. Effet immédiat (les droits sont
  évalués à chaque requête, pas copiés dans l'index).
- **Isolation de la recherche** : les droits sont résolus **en SQL** (propriétaire + partages) en une liste d'identifiants de documents ;
  la requête Chroma est filtrée sur `document_id $in [...]`, l'index BM25 est restreint au même ensemble, et les passages sortants sont
  re-filtrés (défense en profondeur). Une liste vide ne lance aucune recherche. `retrieval.retrieve(..., allowed_ids=...)` n'a pas de valeur
  par défaut : impossible d'oublier le filtre.
- **Déduplication par propriétaire** : l'unicité du contenu est `(owner_id, sha256)`. Un autre utilisateur qui envoie le même fichier obtient
  son propre document (sinon « ce document existe déjà » révélerait que quelqu'un le possède).
- **Limitation de débit** : par utilisateur (`RATE_LIMIT_PER_MINUTE`), via Redis (partagée entre instances), repli en mémoire si Redis tombe ;
  login/inscription limités par IP et par e-mail (`AUTH_RATE_LIMIT_PER_MINUTE`).

## API documents (en-tête `Authorization: Bearer <jeton>`)
| Route | Rôle |
|---|---|
| `POST /documents/upload` | Import PDF, images, DOCX, PPTX, XLSX, TXT, MD (202 ; 200 si contenu déjà présent chez vous) |
| `GET /documents?limit&offset&status&scope` | Vos documents + partagés (`scope=mine`, `scope=all` admin) ; `X-Total-Count` |
| `GET /documents/{id}` · `/text` | Statut · texte extrait |
| `POST /documents/{id}/reprocess` | Relancer un traitement (propriétaire/admin) |
| `DELETE /documents/{id}` | Supprime fichier, texte, vecteurs, partages, ligne SQL (propriétaire/admin) |
| `POST /ask` | `{question, top_k?, profile?, document_ids?, scope?}` -> réponse + sources citées |
| `POST /documents/{id}/summary` | Résumé hiérarchique par sections (map-reduce), mis en cache ; `?refresh=true` (propriétaire/admin) |
| `GET /health` · `/ready` | Liveness · readiness (DB + vectorstore), publics |
| `GET /metrics` | Prometheus (non exposé par Caddy ; `METRICS_TOKEN` optionnel) |

## Extraction (v2.1)
- Texte extrait **page par page** ; chaque chunk porte `page`, `section` (titres détectés par taille de police,
  ou styles Heading/Titre pour DOCX, `#` pour Markdown) et `kind` (`text` | `table` | `figure`).
- **Tableaux** des PDF natifs, DOCX, PPTX, XLSX -> Markdown (en-tête répété dans chaque chunk).
- **Gros documents** : traitement par lots de `PAGE_BATCH_SIZE` pages, progression `pages_done/pages_total`
  dans `GET /documents/{id}`, nettoyage si le document est supprimé ou échoue en cours de route.
- **Figures** : `VLM_PROVIDER=ollama|openai` ajoute une description indexée avec sa page (désactivé par défaut).
  Les graphiques PPTX natifs sont lus directement (données), sans VLM.
- Limites : pas de détection de tableaux/titres sur les pages scannées (OCR brut) ; DOCX : numéros de page
  approximatifs (issus des sauts de page enregistrés par Word) ; formats .doc/.ppt/.xls non supportés.
- Migration : `alembic upgrade head` (0002 ajoute `pages_total`, `pages_done`).

## Indexation et recherche (v2.2)
Un **profil** (`app/services/profiles.py`) = stratégie de chunking + modèle d'embedding + récupération + `top_k`
+ `min_score`, avec sa propre collection Chroma. Plusieurs profils peuvent partager un index
(`e5-fixed` / `e5-hybrid` / `e5-hybrid-rerank`) : seule la recherche change.

- **Chunking** : `fixed` (fenêtres + overlap, confinées à un segment), `page`, `section` (peut franchir une page :
  `page`..`page_end`), `sentence` (phrases entières, overlap en phrases).
- **Embeddings** : MiniLM (témoin, tronque à ~128 tokens), `intfloat/multilingual-e5-base` (préfixes query/passage
  gérés), `BAAI/bge-m3`.
- **Hybride** : dense + BM25 (implémentation maison, même formule que `rank_bm25`, index inversé, filtre par
  document, filtre de couverture des termes) fusionnés par **RRF** (k=60) ; `min_score` filtre la liste dense avant fusion.
- **Reranker** : cross-encoder (`BAAI/bge-reranker-v2-m3`) sur le top 20 ; seuil optionnel `rerank_min_score`.
- **Résumé** : plus de détection par regex ; `POST /documents/{id}/summary` lit `segments.jsonl` (écrit à
  l'ingestion) -> résumé de blocs -> de sections -> global. Documents indexés avant la v2.2 : `/reprocess`.

### Comparer les profils
```bash
# 1. indexer les profils à comparer (une seule extraction OCR pour tous)
INDEX_PROFILES=baseline,minilm-250,e5-fixed,bge-fixed,e5-sentence,e5-section,e5-page
docker compose exec api python -m scripts.reindex
# 2. évaluer (jeu annoté : data/benchmark/questions.example.jsonl)
docker compose exec api python -m scripts.benchmark data/benchmark/questions.jsonl \
    --profiles baseline,minilm-250,e5-fixed,bge-fixed,e5-sentence,e5-section,e5-page,e5-hybrid,e5-hybrid-rerank --out report.json
```
Ordre conseillé : (1) modèles à chunking fixe (`baseline`, `minilm-250`, `e5-fixed`, `bge-fixed`), (2) chunking avec le
meilleur modèle (surcharger via `RAG_PROFILES_JSON`), (3) hybride puis rerank sur le meilleur index.
Calibrer `min_score` avec les questions `"answerable": false` (colonne `refus_corrects`) : les valeurs fournies sont des
points de départ, pas des mesures.

### ChromaDB ou pgvector ? (à trancher avec le benchmark)
Rester sur Chroma si : un seul process suffit, la latence dense p95 est négligeable devant le LLM, le corpus tient en
mémoire. Passer à pgvector si : plusieurs instances/workers sont nécessaires, il faut filtrer par métadonnées SQL ou
jointer avec les tables, une seule base à sauvegarder, ou la recherche plein texte PostgreSQL (`tsvector`) doit
remplacer BM25 en mémoire (qui est reconstruit à chaque modification, process unique). La qualité de retrieval ne
dépend pas du store : seules latence, opérations et passage à l'échelle comptent.

## Exploitation

### Architecture (docker-compose.yml)
`caddy` (HTTPS) -> `api` (FastAPI, 1 process) · `worker` x N (Celery : OCR, chunking, embeddings) · `redis` (file, limiteur, verrous, versions
d'index) · `chroma` (serveur vectoriel partagé) · `db` (PostgreSQL) · `backup` · `prometheus` + `grafana` (profil `monitoring`).
```bash
docker compose up -d --scale worker=3                 # 3 workers d'ingestion (chacun charge ses modèles : prévoir la RAM)
docker compose --profile monitoring up -d             # Prometheus :9090, Grafana :3000 (127.0.0.1 seulement)
```
- **Pourquoi un serveur Chroma** : ChromaDB embarqué n'est pas sûr avec plusieurs process qui écrivent. `QUEUE_BACKEND=celery` exige
  `CHROMA_HOST` (la configuration est refusée sinon). Sans Redis/Celery (`QUEUE_BACKEND=thread`, par défaut), tout reste dans un process.
- **Tâches** : `acks_late` (reprise si un worker meurt), un bail Redis par document (jamais deux workers sur le même document, prolongé à
  chaque page), `visibility_timeout` de 6 h. Au démarrage d'un worker, les documents `PENDING`/`PROCESSING` sont republiés (les doublons
  sont neutralisés : un document déjà `INDEXED` est ignoré).
- **Index BM25** : reconstruit quand le compteur de version de la collection change ; ce compteur est dans Redis, donc un worker qui
  indexe invalide le cache de toutes les instances API.
- **Plusieurs instances API** : `docker compose up -d --scale api=2` ne suffit pas avec Caddy tel quel (une seule adresse résolue à
  l'avance) ; prévoir `dynamic a` dans `reverse_proxy` ou un répartiteur dédié.

### Métriques (`/metrics`, workers : port 9101)
`docai_http_request_duration_seconds` / `docai_http_requests_total` (par gabarit de route), `docai_errors_total{kind}`,
`docai_llm_tokens_total{provider,model,kind}`, `docai_llm_request_duration_seconds`, `docai_retrieval_stage_duration_seconds{stage}`
(embed, dense, bm25_fusion, rerank), `docai_ingestion_documents_total{status}`, `docai_ingestion_duration_seconds`,
`docai_documents{status}` et `docai_queue_length` (file Redis). Un tableau de bord Grafana est provisionné (`monitoring/grafana`).
Journaux : `LOG_FORMAT=json` -> une ligne JSON par requête (`request_id`, `user_id`, `route`, `status`, `duration_ms`).

### Versionnage de l'index
- `INDEX_VERSION` suffixe toutes les collections (`documents_v2` -> `documents_v3`). Pour changer de modèle ou de chunking sans interruption :
  déployer avec `INDEX_VERSION=v3`, lancer `python -m scripts.reindex` (nouvelle collection construite à côté de l'ancienne), vérifier avec
  `scripts.benchmark`, puis supprimer l'ancienne collection. Revenir en arrière = remettre l'ancienne valeur.
- La table `index_collections` mémorise, pour chaque collection, le modèle d'embedding et les paramètres de chunking qui l'ont produite ;
  `documents.index_version` indique la version qui contient chaque document ; `GET /admin/system` expose le tout.

### Tests de charge (Locust)
```bash
pip install -r requirements-loadtest.txt
# serveur : LLM_PROVIDER=mock, AUTH_RATE_LIMIT_PER_MINUTE=0, RATE_LIMIT_PER_MINUTE=0 (mesurer l'infra, pas Groq)
LOAD_VERIFY_TLS=false locust -f loadtests/locustfile.py --host https://localhost --headless -u 30 -r 3 -t 5m --csv out
```
Chaque utilisateur virtuel crée son compte, envoie des .md uniques (pas d'OCR) et pose des questions. Surveiller dans Grafana la taille de
la file, la latence p95 de `/ask` et la durée d'ingestion ; recommencer avec 1, 2 puis 4 workers pour voir le gain. Refaire un passage avec les
limites réelles pour vérifier les 429.

### Sauvegardes
Le service `backup` fait toutes les `BACKUP_INTERVAL_HOURS` : `pg_dump` + archive des fichiers (`uploads`, texte extrait) + somme de contrôle,
dans `./backups/<horodatage>/` (rétention `BACKUP_KEEP_DAYS`). Les vecteurs ne sont pas inclus par défaut (`scripts.reindex` les reconstruit ;
une copie à chaud de Chroma peut être incohérente). Restauration : `./scripts/restore.sh backups/<horodatage>`.
**À faire vous-même** : copier `./backups` hors de la machine (autre disque, S3/rclone) et tester une restauration sur une machine vierge.

## Tests
```bash
pip install -r requirements.txt -r requirements-dev.txt && pytest
```
`tests/test_isolation.py` : A ne voit jamais les documents de B (liste, accès direct, texte, résumé, retraitement, suppression, `/ask` dense et
BM25, même avec un index qui ignorerait le filtre), partages, révocation, admin. `tests/test_auth.py` : JWT, mots de passe, rôles.
`tests/test_ops.py` : limiteur Redis par utilisateur, baux, versions partagées, logs JSON, métriques.

## Limites connues
- Le filtre vectoriel passe par une liste d'identifiants (`document_id $in [...]`) : au-delà de quelques dizaines de milliers de documents
  accessibles par utilisateur, SQLite (Chroma) plafonne le nombre de variables ; c'est le moment de passer à pgvector (filtre SQL natif).
- Pas de quotas de stockage par utilisateur, pas de vérification d'e-mail, pas de réinitialisation de mot de passe, pas de MFA.
- Le serveur Chroma n'est pas authentifié : il ne doit jamais être publié hors du réseau Docker.
- Index BM25 en mémoire, reconstruit à chaque modification de la collection (coût croissant avec le corpus) : pgvector + `tsvector` à terme.
- Les sauvegardes locales ne protègent pas d'une perte de la machine : copie externe indispensable.
