# MODIFIED FILE
# Original source: Genefold/arro-server (https://github.com/Genefold/arro-server)
# Copyright 2026 GENEFOLD AI LTD — Apache License 2.0
# Modifications by Tommaso Moriondo for the LEAF Prompt-Kaban POC:
#   - Added EmbedderService singleton wrapping nomic-embed-text-v1.5
#   - Added lazy-load pattern with SentenceTransformer trust_remote_code
#   - Added task-prefix injection required by the Nomic v1.5 model
# See CHANGES.md for full modification record.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

_TASK_PREFIX = "search_query: "


class EmbedderService:
    """Singleton that wraps a SentenceTransformer model for query embedding.

    Only loaded when the [nlp] extra is installed.  The engine falls back
    gracefully to vector-only search when this service is absent.
    """

    _instance: "EmbedderService | None" = None

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for natural-language search. "
                "Install it with: pip install 'arro-server[nlp]'"
            ) from exc

        log.info("Loading embedder model: %s", model_name)
        self.model_name = model_name
        self._model     = SentenceTransformer(
            model_name,
            trust_remote_code=True,
        )
        EmbedderService._instance = self
        log.info("Embedder model loaded.")

    @classmethod
    def get(cls) -> "EmbedderService":
        if cls._instance is None:
            from .settings import get_settings
            settings = get_settings()
            cls._instance = cls(settings.embedder_model)
        return cls._instance

    def embed(self, text: str) -> np.ndarray:
        """Return a float64 unit-normalised embedding for *text*."""
        prefixed = _TASK_PREFIX + text
        vec = self._model.encode(
            prefixed,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(vec, dtype=np.float64)
