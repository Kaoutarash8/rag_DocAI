class LLMError(Exception):
    """Le fournisseur LLM est indisponible ou a renvoyé une erreur."""


class ProfileError(Exception):
    """Profil de recherche inconnu ou non indexé."""
