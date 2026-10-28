"""Test de charge : uploads concurrents + questions /ask, chaque utilisateur virtuel ayant son compte.

    pip install -r requirements-loadtest.txt
    locust -f loadtests/locustfile.py --host https://localhost            # interface sur :8089
    locust -f loadtests/locustfile.py --host https://localhost --headless -u 30 -r 3 -t 5m --csv out

Préparer le serveur pour mesurer l'infrastructure (et non le LLM ni les limiteurs) :
    LLM_PROVIDER=mock                  # pas d'appel à Groq/OpenAI, pas de coût ni de quota
    AUTH_RATE_LIMIT_PER_MINUTE=0       # sinon l'IP unique de Locust est bloquée dès l'inscription
    RATE_LIMIT_PER_MINUTE=0            # (ou une valeur élevée)
Refaire ensuite UN passage avec les limites réelles pour vérifier que les 429 arrivent bien.

Variables : LOAD_VERIFY_TLS=false (certificat local Caddy), LOAD_PASSWORD, LOAD_INDEX_WAIT_S (attente max
de l'indexation du document initial, 180 par défaut).
Les fichiers envoyés sont des .md (pas d'OCR) : on mesure la file, les embeddings et la base, pas PaddleOCR.
"""
import os
import random
import time
import uuid

from locust import HttpUser, between, task

PASSWORD = os.getenv("LOAD_PASSWORD", "load-test-password-123")
INDEX_WAIT_S = int(os.getenv("LOAD_INDEX_WAIT_S", "180"))
QUESTIONS = [
    "Quel est le sujet principal du document ?", "Quelles dates sont mentionnées ?",
    "Résume la section sur la sécurité.", "Quels sont les risques identifiés ?",
    "Qui est responsable de la mise en oeuvre ?", "Quel est le budget prévu ?",
]
WORDS = "contrat budget sécurité projet livrable client délai risque qualité équipe réunion version".split()


def make_markdown() -> bytes:
    """Contenu unique à chaque appel (sinon la déduplication renverrait 200 sans traitement)."""
    body = " ".join(random.choice(WORDS) for _ in range(400))
    return f"# Rapport {uuid.uuid4().hex}\n\n## Introduction\n{body}\n\n## Détails\n{body[::-1]}\n".encode()


class DocUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.email = f"load-{uuid.uuid4().hex[:12]}@loadtest.local"
        self.client.verify = os.getenv("LOAD_VERIFY_TLS", "true").lower() != "false"
        with self.client.post("/auth/register", json={"email": self.email, "password": PASSWORD},
                              name="/auth/register", catch_response=True) as r:
            if r.status_code != 201:
                r.failure(f"inscription impossible : {r.status_code} {r.text[:100]}")
                self.environment.runner.quit()
                return
        tok = self.client.post("/auth/login", json={"email": self.email, "password": PASSWORD},
                               name="/auth/login").json()["access_token"]
        self.client.headers["Authorization"] = f"Bearer {tok}"
        self.doc_ids: list[int] = []
        first = self._upload(name="/documents/upload (initial)")
        deadline = time.time() + INDEX_WAIT_S  # /ask n'a de sens qu'avec au moins un document indexé
        while first and time.time() < deadline:
            r = self.client.get(f"/documents/{first}", name="/documents/{id} (attente)")
            if r.ok and r.json()["status"] in ("indexed", "failed"):
                break
            time.sleep(2)

    def _upload(self, name="/documents/upload"):
        with self.client.post("/documents/upload", name=name, catch_response=True,
                              files={"file": (f"rapport-{uuid.uuid4().hex[:6]}.md", make_markdown(), "text/markdown")}) as r:
            if r.status_code == 202:
                doc_id = r.json()["id"]
                self.doc_ids.append(doc_id)
                return doc_id
            r.failure(f"upload : {r.status_code}")

    @task(6)
    def ask(self):
        with self.client.post("/ask", json={"question": random.choice(QUESTIONS)}, name="/ask",
                              catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"/ask : {r.status_code}")

    @task(2)
    def list_documents(self):
        self.client.get("/documents", name="/documents")

    @task(1)
    def upload(self):
        self._upload()

    @task(1)
    def poll_status(self):
        if self.doc_ids:
            self.client.get(f"/documents/{random.choice(self.doc_ids)}", name="/documents/{id}")
