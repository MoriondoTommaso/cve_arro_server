#!/usr/bin/env python3
"""
get_query_emb_nomic.py


Encodes benchmark queries from benchmark_queries_01.json
with nomic-ai/nomic-embed-text-v1.5.


Output shape per file: (1379, 3, d)
  axis 0  →  benchmark entries (1379)
  axis 1  →  profiles:
               [0]  q_medium    short keyword query
               [1]  q_sentence  natural sentence query
               [2]  q_verbose   detailed verbose query
  axis 2  →  embedding dims (768 / 512 / 256)


Output files:
    queries_emb_nomic_768d.npy
    queries_emb_nomic_512d.npy
    queries_emb_nomic_256d.npy


Usage:
    python get_query_emb_nomic.py           # all dims
    python get_query_emb_nomic.py --dims 768
"""


import argparse
import json
import time
from pathlib import Path


import numpy as np
from sentence_transformers import SentenceTransformer


# ── CONFIGURE YOUR PATHS HERE ─────────────────────────────────────────────────
ROOT        = Path("/content/drive/MyDrive")
OUTPUT_DIR  = ROOT / "data" / "raw"
BENCHMARK_PATH = ROOT / "data" / "benchmark_queries_01.json"


MODEL_ID     = "nomic-ai/nomic-embed-text-v1.5"
QUERY_PREFIX = "search_query: "
BATCH_SIZE   = 512
ALL_DIMS     = [768, 512, 256]


# axis 1 order — fixed, must match your eval script index convention
# field names are used DIRECTLY from entry["queries"][field]
PROFILES = [
    ("q_medium",   "q_medium"),    # axis 1 = 0 — short keyword query
    ("q_sentence", "q_sentence"),  # axis 1 = 1 — natural sentence query
    ("q_verbose",  "q_verbose"),   # axis 1 = 2 — detailed verbose query
]



# ── encoding ──────────────────────────────────────────────────────────────────


def encode_queries(model: SentenceTransformer, texts: list[str], label: str) -> np.ndarray:
    """Encode → raw float32 (n, 768), no L2 normalisation."""
    t0 = time.perf_counter()
    print(f"    [{label}] Encoding {len(texts):,} queries...", flush=True)
    vecs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=False,   # RAW embeddings for ArrowSpace
        convert_to_numpy=True,
    )
    print(f"    Done in {time.perf_counter() - t0:.1f}s   shape={vecs.shape}", flush=True)
    return vecs.astype(np.float32)


def matryoshka_slice(vecs: np.ndarray, d: int) -> np.ndarray:
    """
    Take first d dimensions WITHOUT renormalising.
    ArrowSpace benefits from preserving original norms; slicing is enough.
    """
    return vecs[:, :d].astype(np.float32, copy=True)



# ── CLI + main ────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dims", default="all", help="768 | 512 | 256 | all")
    return p.parse_args()



def main() -> None:
    args      = parse_args()
    dims_list = ALL_DIMS if args.dims == "all" else [int(args.dims)]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {BENCHMARK_PATH.name} ...", flush=True)
    with open(BENCHMARK_PATH) as f:
        bq: list[dict] = json.load(f)
    n = len(bq)
    print(f"  {n:,} entries")

    # validate fields exist in first entry
    first_q = bq[0]["queries"]
    for _, field in PROFILES:
        assert field in first_q, f"Field '{field}' not found in queries. Got: {list(first_q.keys())}"
    print(f"  Query fields validated: {[f for _, f in PROFILES]}")
    print(f"  tier2 includes tier1: {bq[0]['relevant_pos']['tier1'][0] in bq[0]['relevant_pos']['tier2']}\n")

    print(f"Loading {MODEL_ID} ...", flush=True)
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
    print("  Model ready.\n")

    # encode all 3 profiles once at 768d, slice per dim
    print("Encoding 3 query profiles at 768d (RAW, no L2 norm) ...", flush=True)
    profile_vecs: dict[str, np.ndarray] = {}
    for pname, field in PROFILES:
        texts = [QUERY_PREFIX + entry["queries"][field] for entry in bq]
        profile_vecs[pname] = encode_queries(model, texts, label=f"{pname}")

    print()

    for d in dims_list:
        arr = np.zeros((n, len(PROFILES), d), dtype=np.float32)
        for pi, (pname, _) in enumerate(PROFILES):
            arr[:, pi, :] = matryoshka_slice(profile_vecs[pname], d)

        out = OUTPUT_DIR / f"queries_emb_nomic_{d}d.npy"
        np.save(out, arr)
        print(f"  ✓  {out.name}   shape={arr.shape}   dtype={arr.dtype}")

    print("\nAll query embeddings done.\n")
    print("Profile axis mapping:")
    for pi, (pname, field) in enumerate(PROFILES):
        print(f"  arr[:, {pi}, :]  →  {pname}  (field: '{field}')")
    print()
    print("NOTE: tier2 in this benchmark does NOT include the tier1 doc.")
    print("      Adjust your recall_tier2 denominator accordingly in the eval script.")



if __name__ == "__main__":
    main()