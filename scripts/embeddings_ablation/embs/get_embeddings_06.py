"""
get_embeddings.py
Builds corpus embeddings using one of the available extraction variants.

Usage:
    python scripts/get_embeddings.py              # builds v4 (recommended)
    python scripts/get_embeddings.py --variant v1
    python scripts/get_embeddings.py --variant v2
    python scripts/get_embeddings.py --variant v5
    python scripts/get_embeddings.py --variant all
"""

from __future__ import annotations

import argparse
import re                                              # ← was missing
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from semantic_engine_demo.json_load import load_json

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
DATA_PATH = DATA_DIR / "dataset.json"

MODEL_NAME = "perplexity-ai/pplx-embed-v1-0.6b"
BATCH_SIZE = 8


# ══════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _clean(text: str) -> str:
    """Strip {{placeholder}} tokens and collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"\{\{[^}]+\}\}", "", text)).strip()


def _is_weak_title(element: dict) -> bool:
    """
    True when the title carries little discriminative signal:
      - 4 words or fewer, OR
      - none of the doc's own tags/category appear in the title
    """
    title       = element["title"].strip()
    title_lower = title.lower()
    title_words = len(title.split())

    signals = (
        element.get("tags", [])
        + [element.get("category", "")]
        + [element.get("subcategory", "")]
    )
    has_signal = any(
        s.lower().replace("-", " ") in title_lower
        for s in signals if s
    )
    return title_words <= 4 or not has_signal


# ══════════════════════════════════════════════════════════════════════════
# EXTRACTION VARIANTS
# ══════════════════════════════════════════════════════════════════════════

def extract_v1(element: dict) -> str:
    """Baseline: structured fields + full raw content."""
    placeholders_str = (
        f"Template variables: {', '.join(element['placeholders'])}."
        if element.get("has_placeholders") and element.get("placeholders")
        else "No template variables."
    )
    return (
        f"Title: {element['title']}\n"
        f"Category: {element['category']} > {element['subcategory']}\n"
        f"Difficulty: {element['difficulty']}\n"
        f"Target model: {element.get('target_model', 'general')}\n"
        f"Tags: {', '.join(element['tags'])}\n"
        f"{placeholders_str}\n\n"
        f"Content:\n{element['content']}"
    )


def extract_v2(element: dict) -> str:
    """Signal-first: placeholders stripped, content truncated, noise fields removed."""
    tags       = ", ".join(element["tags"])
    clean_body = _clean(element["content"])[:300]
    cat        = f"{element['category']} {element['subcategory']}"
    return (
        f"{element['title']}. "
        f"{cat}. "
        f"{tags}. "
        f"{clean_body}"
    )


def extract_v4(element: dict) -> str:
    """
    Adaptive — recommended.
    Weak/vague titles → content-led representation.
    Strong titles     → title-led representation.
    Placeholders stripped. Difficulty/target_model removed.
    """
    title      = element["title"].strip()
    clean_body = _clean(element["content"])
    tags       = ", ".join(element["tags"])
    cat        = f"{element['category']} > {element['subcategory']}"

    if _is_weak_title(element):
        return f"{cat}. {tags}. {clean_body[:350]}"
    else:
        return f"{title}. {cat}. {tags}. {clean_body[:250]}"


def extract_v5(element: dict) -> str:
    """
    Natural language passage style.
    Targets pplx-embed asymmetric (query, passage) training format.
    """
    title      = element["title"].strip()
    clean_body = _clean(element["content"])
    tags       = ", ".join(element["tags"])
    cat        = f"{element['category']}, {element['subcategory']}"

    header = (
        f"A {element['difficulty']}-level prompt for {cat}."
        if _is_weak_title(element)
        else f"{title}. A {element['difficulty']}-level prompt for {cat}."
    )
    return f"{header} Topics: {tags}. {clean_body[:280]}"


# ── Variant registry ───────────────────────────────────────────────────────
VARIANTS: dict[str, dict] = {
    "v1": {
        "fn"    : extract_v1,
        "output": DATA_DIR / "embeddings_v1.npy",
        "notes" : "Baseline. Structured fields + full raw content.",
    },
    "v2": {
        "fn"    : extract_v2,
        "output": DATA_DIR / "embeddings_v2.npy",
        "notes" : "Signal-first. Placeholders stripped, content@300.",
    },
    "v4": {
        "fn"    : extract_v4,
        "output": DATA_DIR / "embeddings_v4.npy",
        "notes" : "Adaptive. Content-forward for weak/vague titles. Recommended.",
    },
    "v5": {
        "fn"    : extract_v5,
        "output": DATA_DIR / "embeddings_v5.npy",
        "notes" : "Natural language passage style for pplx-embed.",
    },
}


# ══════════════════════════════════════════════════════════════════════════
# BUILD
# ══════════════════════════════════════════════════════════════════════════

def build(variant: str, corpus: list[dict], model: SentenceTransformer) -> None:
    cfg       = VARIANTS[variant]
    extract   = cfg["fn"]
    out_path  = cfg["output"]

    print(f"\n── {variant}: {cfg['notes']}")

    # Weak-title diagnostic
    n_weak = sum(_is_weak_title(doc) for doc in corpus)
    print(f"   Weak titles: {n_weak}/{len(corpus)} ({100*n_weak/len(corpus):.1f}%)")

    # Extract
    sentences = [
        extract(doc)
        for doc in tqdm(corpus, desc=f"   extract_{variant}", ncols=80)
    ]

    # Encode
    embeddings = model.encode(
        sentences,
        batch_size           = BATCH_SIZE,
        show_progress_bar    = True,
        normalize_embeddings = True,   # ← required for cosine sim in ArrowSpace
        convert_to_numpy     = True,
    )

    # Save as float64
    embs = np.array(embeddings, dtype=np.float64)
    np.save(out_path, embs)
    print(f"   Saved → {out_path}  shape={embs.shape}")


def main(variants_to_build: list[str]) -> None:
    print("Loading dataset...")
    corpus = load_json(DATA_PATH)
    print(f"  {len(corpus)} documents\n")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(
        MODEL_NAME,
        device            = "cpu",
        trust_remote_code = True,
    )

    for variant in variants_to_build:
        build(variant, corpus, model)

    print("\n── Done ──────────────────────────────────────────────")
    for v in variants_to_build:
        print(f"  {v}  →  {VARIANTS[v]['output'].name}")


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices = list(VARIANTS.keys()) + ["all"],
        default = "v4",                    # ← recommended default
        help    = "Extraction variant to build (default: v4)",
    )
    args = parser.parse_args()

    to_build = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]
    main(to_build)