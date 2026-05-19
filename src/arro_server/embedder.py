# MODIFIED FILE
# Original source: Genefold/arro-server (https://github.com/Genefold/arro-server)
# Copyright 2026 GENEFOLD AI LTD — Apache License 2.0
# Modifications by Tommaso Moriondo for the LEAF Prompt-Kaban POC:
#   - Added EmbedderService singleton wrapping a SentenceTransformer model
#   - Added lazy-load pattern
# Fix: switched default model to all-MiniLM-L6-v2 (384-dim) to match corpus.
#   - Removed nomic-specific trust_remote_code=True (not needed / breaks MiniLM)
#   - Removed 'search_query: ' task prefix (nomic-only; corrupts MiniLM queries)
#   - Added _EXPECTED_DIM guard to catch model/corpus dim mismatch at startup
# See CHANGES.md for full modification record.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# Dimensionality produced by all-MiniLM-L6-v2 and expected by CveSearchEngine.
# If someone overrides ARRO_SERVER_EMBEDDER_MODEL with a different model,
# embed() will raise early with a clear message instead of a cryptic 422 later.
_EXPECTED_DIM = 384


class EmbedderService:
    """Singleton that wraps a SentenceTransformer model for query embedding.

    Default model: sentence-transformers/all-MiniLM-L6-v2 (384-dim).
    Override via ARRO_SERVER_EMBEDDER_MODEL env var — must produce 384-dim
    vectors to match the CVE corpus.

    Only loaded when the [nlp] extra is installed.  Routes fall back
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
        # trust_remote_code=True was required by nomic-embed-text-v1.5 only.
        # all-MiniLM-L6-v2 is a standard ST model — no remote code needed.
        self._model = SentenceTransformer(model_name)
        EmbedderService._instance = self
        log.info("Embedder model loaded: %s", model_name)

    @classmethod
    def get(cls) -> "EmbedderService":
        if cls._instance is None:
            from .settings import get_settings
            settings = get_settings()
            cls._instance = cls(settings.embedder_model)
        return cls._instance

    def embed(self, text: str) -> np.ndarray:
        """Return a float64 unit-normalised 384-dim embedding for *text*.

        Raises ValueError if the loaded model produces an unexpected dimension
        so mismatches are caught early with a clear message.
        """
        # all-MiniLM-L6-v2 needs no task prefix — plain text is correct.
        vec = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = np.asarray(vec, dtype=np.float64)
        if arr.shape[0] != _EXPECTED_DIM:
            raise ValueError(
                f"Embedder model '{self.model_name}' produced {arr.shape[0]}-dim vector "
                f"but the CVE corpus expects {_EXPECTED_DIM}-dim. "
                "Set ARRO_SERVER_EMBEDDER_MODEL to a 384-dim model "
                "(e.g. sentence-transformers/all-MiniLM-L6-v2)."
            )
        return arr
