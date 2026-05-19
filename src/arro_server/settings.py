# MODIFIED FILE
# Original source: Genefold/arro-server (https://github.com/Genefold/arro-server)
# Copyright 2026 GENEFOLD AI LTD — Apache License 2.0
# Modifications by Tommaso Moriondo for the LEAF Prompt-Kaban POC:
#   - Added `prompt_data_dir` field for LEAF Kaban data volume path
#   - Added `embedder_model` field for HuggingFace model id override
# Modifications for CVE spectral drift demo:
#   - Added `cve_period_a` / `cve_period_b` fields pointing at the two
#     embedding slices used by CveDriftEngine (/api/drift/* routes)
#   - Added `cve_n_sample` field controlling subsampling in the drift engine
#   - FIX: cve_period_a/b defaults are now relative path strings resolved at
#     access time, not absolute paths computed from __file__ at import time.
#     The old approach broke in installed / containerised environments where
#     __file__ points inside site-packages.
# See CHANGES.md for full modification record.
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Runtime configuration.

    Environment variables are prefixed with ``ARRO_SERVER_``.

    ``data_roots`` accepts a comma-separated list, e.g.
    ``ARRO_SERVER_DATA_ROOTS=/data/zarr,/mnt/shared/zarr``.
    Each root may optionally be prefixed with a label: ``label=path``.

    ``cors_origins`` accepts a comma-separated list of allowed origins, e.g.
    ``ARRO_SERVER_CORS_ORIGINS=https://app.example.com,https://admin.example.com``.
    Use ``*`` (the default) to allow all origins — do not use ``*`` in production.

    CVE drift paths
    ---------------
    ``cve_period_a`` / ``cve_period_b`` default to relative paths
    ``data/cve_embeddings_demo/embs_99_to_14.npy`` and
    ``data/cve_embeddings_demo/embs_15_to_2025.npy`` resolved against the
    process CWD at runtime.  Override via environment variables
    ``ARRO_SERVER_CVE_PERIOD_A`` / ``ARRO_SERVER_CVE_PERIOD_B`` (absolute or
    relative paths).

    ``cve_n_sample`` controls how many rows are passed to ArrowSpaceBuilder.
    Defaults to 8 000; set to 0 to disable subsampling (slow on large arrays).
    """

    model_config = SettingsConfigDict(
        env_prefix="ARRO_SERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_roots: Annotated[list[str], NoDecode] = Field(default_factory=list)
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])
    default_window: int = 100
    max_window: int = 10_000
    serve_frontend: bool = True
    frontend_dir: str | None = None
    index_store: str = "./arrowspace_index"
    index_cache_size: int = 8

    # ── Prompt search (LEAF Kaban) ──────────────────────────────────────────────
    prompt_data_dir: str = "./data"
    embedder_model: str = "nomic-ai/nomic-embed-text-v1.5"

    # ── CVE spectral drift (two-period demo) ────────────────────────────────────
    # Relative to CWD so they work both locally (run from repo root) and inside
    # Docker (WORKDIR /app with data mounted at /app/data).
    cve_period_a: str = "data/cve_embeddings_demo/embs_99_to_14.npy"
    cve_period_b: str = "data/cve_embeddings_demo/embs_15_to_2025.npy"
    cve_n_sample: int = 8_000   # rows subsampled per period for the index build

    @field_validator("data_roots", "cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @property
    def resolved_roots(self) -> dict[str, Path]:
        """Return mapping of root label -> filesystem path.

        Roots may be specified as ``path`` or ``label=path``.
        Unlabeled roots are auto-named after their directory basename, with
        numeric suffixes on collision.

        Implemented as a plain ``@property`` (not ``cached_property``) to
        avoid the pydantic-settings v2 incompatibility where ``cached_property``
        descriptors are not stored in ``__dict__`` and can raise
        ``AttributeError`` on some versions.
        """
        out: dict[str, Path] = {}
        for entry in self.data_roots:
            if "=" in entry:
                label, raw = entry.split("=", 1)
                label = label.strip()
            else:
                raw = entry
                label = Path(raw).name or "root"
            path = Path(raw).expanduser().resolve()
            base = label
            i = 1
            while label in out:
                i += 1
                label = f"{base}-{i}"
            out[label] = path
        return out

    def warn_insecure_defaults(self) -> None:
        """Log a warning if CORS is open to all origins."""
        if "*" in self.cors_origins:
            log.warning(
                "SECURITY: ARRO_SERVER_CORS_ORIGINS is set to '*' (allow all). "
                "This is unsafe in production — set it to your frontend origin(s)."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton."""
    s = Settings()
    s.warn_insecure_defaults()
    return s


def reset_settings_cache() -> None:
    """Test / reload helper — clears the lru_cache on get_settings."""
    get_settings.cache_clear()
