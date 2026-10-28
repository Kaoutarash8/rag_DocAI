"""Description des figures (graphiques, schémas, photos) par un modèle vision (VLM).

Désactivé par défaut (VLM_PROVIDER=none). Un échec n'interrompt jamais l'ingestion :
la figure est simplement ignorée.
"""

import base64
import logging

from app.core.config import get_settings

log = logging.getLogger(__name__)

PROMPT = (
    "Tu analyses une figure extraite d'un document d'entreprise. Réponds en français, "
    "en 120 mots maximum, sans inventer. Indique : 1) le type de figure (graphique en barres, "
    "courbe, schéma d'architecture, organigramme, photo, logo...), 2) son titre ou sa légende "
    "s'ils sont visibles, 3) les axes, unités et séries pour un graphique, 4) les valeurs et "
    "tendances clés, ou pour un schéma les éléments et leurs relations, 5) tout texte lisible. "
    "Si l'image est purement décorative (logo, fond, icône), réponds exactement : DECORATIF."
    "{context}"
)


def enabled() -> bool:
    return get_settings().vlm_provider != "none"


def describe(image: bytes, mime: str = "image/png", context: str = "") -> str | None:
    """Retourne la description, ou None (désactivé, décoratif, erreur)."""
    s = get_settings()
    if s.vlm_provider == "none":
        return None
    prompt = PROMPT.format(context=f"\nContexte (section du document) : {context}" if context else "")
    try:
        if s.vlm_provider == "ollama":
            import ollama

            client = ollama.Client(host=s.ollama_base_url, timeout=s.vlm_timeout_s)
            r = client.chat(model=s.vlm_model, options={"num_predict": 300},
                            messages=[{"role": "user", "content": prompt, "images": [image]}])
            text = r["message"]["content"]
        else:  # openai (ou tout serveur compatible via OPENAI_BASE_URL)
            from openai import OpenAI

            if not s.openai_api_key:
                log.warning("VLM openai sans OPENAI_API_KEY : figure ignorée")
                return None
            client = OpenAI(api_key=s.openai_api_key, timeout=s.vlm_timeout_s, max_retries=1)
            url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
            r = client.chat.completions.create(
                model=s.vlm_model, max_tokens=300, temperature=0.1,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": url}}]}])
            text = r.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001 - ne jamais faire échouer le document pour une figure
        log.warning("Description de figure impossible : %s", e)
        return None
    text = text.strip()
    if not text or text.upper().startswith("DECORATIF"):
        return None
    return text
