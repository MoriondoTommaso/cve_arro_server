"""
get_query_emb.py
Generates query embeddings from queries.json using nomic-embed-text-v1.5.

Outputs:
  data/raw/queries_emb_768.npy    → (n_entries, 768)
  data/raw/queries_emb_512.npy    → (n_entries, 512)
  data/raw/queries_emb_256.npy    → (n_entries, 256)
  data/raw/queries_index.json     → mapping: query_id → {row_index, expected_prompt_id, query_text}
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# --- Configuration ---
ROOT         = Path("/content/drive/MyDrive")
OUTPUT_DIR   = ROOT / "data" / "raw"
QUERY_PATH   = ROOT / "data" / "queries.json"
OUT_INDEX    = OUTPUT_DIR / "queries_index.json"

MODEL_ID     = "nomic-ai/nomic-embed-text-v1.5"
QUERY_PREFIX = "search_query: "  # Required by Nomic for queries
BATCH_SIZE   = 512               # Increased for GPU processing
ALL_DIMS     = [768, 512, 256]   # Matryoshka dimensions

def main():
    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading queries from {QUERY_PATH}...")
    with open(QUERY_PATH, "r") as f:
        query_corpus = json.load(f)
        
    n = len(query_corpus)
    print(f"  Loaded {n} queries.")

    sentences = []
    index_mapping = {}

    for idx, entry in enumerate(query_corpus):
        query_id = entry["query_id"]
        query_text = entry["query_text"]
        expected_id = entry["expected_prompt_id"]
        
        # Apply the Nomic specific query prefix
        prefixed_text = QUERY_PREFIX + query_text
        sentences.append(prefixed_text)
        
        index_mapping[query_id] = {
            "row_index": idx,
            "expected_prompt_id": expected_id,
            "query_text": query_text
        }

    print(f"Loading model: {MODEL_ID}")
    # Removed device="cpu" so Colab automatically uses the T4 GPU
    model = SentenceTransformer(
        MODEL_ID, 
        trust_remote_code=True
    )

    # ── Encode queries ──────────────────────────────────────────
    print(f"Encoding {len(sentences)} queries with batch size {BATCH_SIZE}...")
    embeddings = model.encode(sentences, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
    
    # ── Save Embeddings (Matryoshka Truncation) ─────────────────
    # Nomic natively supports slicing the trailing dimensions to reduce vector size
    for dim in ALL_DIMS:
        print(f"Processing dimension: {dim}...")
        # Slice to the target dimension
        emb_sliced = embeddings[:, :dim]
        
        # Re-normalize the truncated embeddings for accurate cosine similarity
        emb_normalized = emb_sliced / np.linalg.norm(emb_sliced, axis=1, keepdims=True)
        emb_normalized = emb_normalized.astype(np.float32) # float32 is standard for Vector DBs
        
        out_emb_path = OUTPUT_DIR / f"queries_emb_{dim}.npy"
        np.save(out_emb_path, emb_normalized)
        print(f"  Saved → {out_emb_path}  shape={emb_normalized.shape}")

    # ── Save query index mapping ─────────────────────────────────
    with open(OUT_INDEX, "w") as f:
        json.dump(index_mapping, f, indent=4)
    print(f"  Saved → {OUT_INDEX}")

    print("\nDone. Ready for evaluation!")


if __name__ == "__main__":
    main()