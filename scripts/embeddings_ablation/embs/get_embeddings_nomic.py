#!/usr/bin/env python3
"""
get_embeddings_nomic.py

Generates corpus embeddings for dataset.json using nomic-ai/nomic-embed-text-v1.5.
Produces 9 .npy files: 3 extraction variants × 3 Matryoshka dimensions.

Output files (adjust paths to match your layout):
    embeddings_nomic_compact_768d.npy     shape (20000, 768)
    embeddings_nomic_compact_512d.npy     shape (20000, 512)
    embeddings_nomic_compact_256d.npy     shape (20000, 256)
    embeddings_nomic_full_768d.npy        shape (20000, 768)
    embeddings_nomic_full_512d.npy        shape (20000, 512)
    embeddings_nomic_full_256d.npy        shape (20000, 256)
    embeddings_nomic_structured_768d.npy  shape (20000, 768)
    embeddings_nomic_structured_512d.npy  shape (20000, 512)
    embeddings_nomic_structured_256d.npy  shape (20000, 256)

Usage:
    python get_embeddings_nomic.py                        # all 9
    python get_embeddings_nomic.py --variant compact      # one variant, all dims
    python get_embeddings_nomic.py --variant full --dims 512
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path("/content/drive/MyDrive")
DATASET_PATH = ROOT / "data" / "dataset.json"
OUTPUT_DIR   = ROOT / "data"

MODEL_ID   = "nomic-ai/nomic-embed-text-v1.5"
DOC_PREFIX = "search_document: "
BATCH_SIZE = 256
ALL_DIMS   = [768, 512, 256]


# ── helpers ───────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    """Remove {{placeholder}} tokens, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"\{\{[^}]+\}\}", "", text)).strip()


def is_weak_title(el: dict) -> bool:
    """True when title has ≤4 words or shares no token with its own tags/category."""
    t = el["title"].lower()
    signals = el.get("tags", []) + [el.get("category", ""), el.get("subcategory", "")]
    has_signal = any(s.lower().replace("-", " ") in t for s in signals if s)
    return len(el["title"].split()) <= 4 or not has_signal


# ── extraction variants ───────────────────────────────────────────────────────

def extract_compact(el: dict) -> str:
    """
    Signal-first, 350-char body cap.
    Category + tags front-loaded  →  early Matryoshka dims (0-255) receive
    clean categorical signal; document content populates discriminative later dims.
    Best baseline: mirrors the v2 structure that won the previous sweep.
    """
    title = el["title"].strip()
    tags  = ", ".join(el["tags"])
    body  = clean(el["content"])[:350]
    cat   = f"{el['category']} > {el.get('subcategory', '')}"

    if is_weak_title(el):
        return DOC_PREFIX + f"{cat}. {tags}. {body}"
    return DOC_PREFIX + f"{title}. {cat}. {tags}. {body}"


def extract_full(el: dict) -> str:
    """
    Full content, zero truncation.
    Exploits nomic's 8192-token RoPE window.
    Late Matryoshka dims (512-767) absorb fine-grained document texture
    →  primary fix for Type-B failures (coding 404 docs, audio-engineering 155,
       architecture 86): finds neighbourhood but can't rank exact doc first.
    """
    title = el["title"].strip()
    tags  = ", ".join(el["tags"])
    body  = clean(el["content"])
    cat   = f"{el['category']} > {el.get('subcategory', '')}"
    diff  = el.get("difficulty", "")

    return DOC_PREFIX + f"{title}. {cat}. Difficulty: {diff}. Tags: {tags}. {body}"


def extract_structured(el: dict) -> str:
    """
    Richest form: explicit field labels + full content.
    Named fields (Title:, Category:, Tags:) guide nomic attention to distribute
    semantic load evenly across ALL Matryoshka bands.
    Primary fix for Type-A failures (branding, coaching, communication, audit):
    engine is completely blind — needs every contextual signal available.
    """
    title  = el["title"].strip()
    tags   = ", ".join(el["tags"])
    body   = clean(el["content"])
    cat    = el["category"]
    subcat = el.get("subcategory", "")
    diff   = el.get("difficulty", "general")
    ph     = el.get("placeholders", [])
    ph_str = f"Variables: {', '.join(ph)}.\n" if ph else ""

    return (
        DOC_PREFIX
        + f"Title: {title}\n"
          f"Category: {cat} > {subcat}\n"
          f"Difficulty: {diff}\n"
          f"Tags: {tags}\n"
        + ph_str
        + body
    ).strip()


VARIANTS = {
    "compact":    extract_compact,
    "full":       extract_full,
    "structured": extract_structured,
}


# ── encoding ──────────────────────────────────────────────────────────────────

def encode_corpus(model: SentenceTransformer, texts: list[str], label: str) -> np.ndarray:
    """Encode → L2-normalised float32 array of shape (n, 768)."""
    t0 = time.perf_counter()
    print(f"  Encoding {len(texts):,} documents [{label}]...", flush=True)
    vecs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s   shape={vecs.shape}", flush=True)
    return vecs.astype(np.float32)


def matryoshka_slice(vecs: np.ndarray, d: int) -> np.ndarray:
    """
    Take first d dimensions and re-normalise.
    Safe because MRL trains each prefix independently —
    slicing + renorm == training at that dimension.
    NOTE: caller must cast to float64 before passing to ArrowSpaceBuilder.
    """
    sl    = vecs[:, :d].copy()
    norms = np.linalg.norm(sl, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (sl / norms).astype(np.float32)


# ── CLI + main ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant",
        choices=list(VARIANTS) + ["all"],
        default="all",
        help="Which extraction variant to run (default: all)",
    )
    p.add_argument(
        "--dims",
        default="all",
        help="Matryoshka dims to produce: 768 | 512 | 256 | all (default: all)",
    )
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    dims_list = ALL_DIMS if args.dims == "all" else [int(args.dims)]
    var_list  = list(VARIANTS) if args.variant == "all" else [args.variant]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {DATASET_PATH} ...", flush=True)
    with open(DATASET_PATH) as f:
        corpus: list[dict] = json.load(f)
    print(f"  {len(corpus):,} documents loaded\n")

    print(f"Loading {MODEL_ID} ...", flush=True)
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
    print("  Model ready.\n")

    for vname in var_list:
        fn = VARIANTS[vname]

        print(f"{'━' * 60}")
        print(f"  VARIANT: nomic_{vname}")
        print(f"{'━' * 60}")

        texts = [fn(doc) for doc in corpus]

        # sanity-check first document
        print(f"  Sample (first doc):\n  {texts[0][:160]!r}\n")

        # encode once at full 768d — then slice for each requested dim
        raw_vecs = encode_corpus(model, texts, label=f"nomic_{vname}")

        for d in dims_list:
            vd  = matryoshka_slice(raw_vecs, d)
            out = OUTPUT_DIR / f"embeddings_nomic_{vname}_{d}d.npy"
            np.save(out, vd)
            print(f"  ✓  Saved: {out.name}   shape={vd.shape}   dtype={vd.dtype}")

        print()

    print("All corpus embeddings done.")
    print()
    print("IMPORTANT — when loading into ArrowSpaceBuilder always cast:")
    print("  embs = np.load('embeddings_nomic_compact_768d.npy').astype(np.float64)")


if __name__ == "__main__":
    main()