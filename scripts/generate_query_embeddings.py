#!/usr/bin/env python3
"""
generate_query_embeddings.py

Generates query embeddings for the 40 test queries using all the models
that produced the corpus embeddings, then saves them to disk.

Corpus → Model mapping (inferred from shapes):
  nomic_256 / nomic_512 / nomic_768  →  nomic-ai/nomic-embed-text-v1.5   (Matryoshka, truncate dim)
  emb_02 / emb_05                    →  sentence-transformers/all-MiniLM-L6-v2  (384d)
  emb_v1 / emb_v2 / emb_v4 /
    emb_v5 / emb_06                  →  llmrails/ember-v1  (1024d)  ← "perplexity 1024"
                                        (alt: intfloat/e5-large-v2 also 1024d — swap MODEL_1024 if needed)

Outputs (saved next to the corpus files):
  data/data_raw/benchmarks/query_embs_nomic/queries_emb_nomic_256d.npy
  data/data_raw/benchmarks/query_embs_nomic/queries_emb_nomic_512d.npy
  data/data_raw/benchmarks/query_embs_nomic/queries_emb_nomic_768d.npy
  data/data_raw/benchmarks/query_embs/queries_emb_384.npy      (all-MiniLM)
  data/data_raw/benchmarks/query_embs/queries_emb_1024.npy     (ember / perplexity)
"""

from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).parent.parent
BENCH = ROOT / "data" / "data_raw" / "benchmarks"
OUT_NOMIC = BENCH / "query_embs_nomic"
OUT_STD   = BENCH / "query_embs"

OUT_NOMIC.mkdir(parents=True, exist_ok=True)
OUT_STD.mkdir(parents=True, exist_ok=True)


test_queries = [
    # Semantic & Intent-Based
    "I'm having trouble getting my team to adopt the new CRM software, looking for change management tips.",
    "What's the best way to gently tell a client their budget is completely unrealistic?",
    "Our churn rate spiked last month and I need a framework to investigate why.",
    "Looking for ways to improve employee retention without just throwing money at the problem.",
    "I need help mediating a conflict between two senior engineers who strongly disagree on the system architecture.",
    "How can I pitch a high-risk, high-reward marketing campaign to a conservative board of directors?",
    "My startup is running out of runway, what are the immediate steps I should take?",
    "I want to transition from a technical contributor to an engineering manager, what skills am I missing?",
    "We need to re-engage dormant users who haven't logged in for over six months.",
    "Help me figure out why our recent product launch didn't generate the buzz we expected.",
    # Conversational & Colloquial
    "Is it legal for my landlord to keep my deposit for normal wear and tear in New York?",
    "My React app is super laggy on mobile, what am I doing wrong?",
    "Can someone explain Docker to me like I'm 5?",
    "I totally bombed my interview, how do I write a follow-up email that doesn't sound desperate?",
    "Why does my Wi-Fi keep dropping only when I'm on Zoom calls?",
    "Is there a quick way to clean up this messy Excel file without doing it manually?",
    "What happens if I accidentally put regular gas in a premium car?",
    "I think my coworker is stealing my ideas, how do I bring this up to HR?",
    "How do I fix a leaky faucet in my kitchen without calling a plumber?",
    "My dog keeps eating grass and throwing up, should I be worried?",
    # Highly Specific & Technical
    "Configuring BGP routing policies for multi-homed AS with unequal bandwidth links.",
    "Optimizing garbage collection pauses in Java 17 using ZGC for low-latency trading applications.",
    "Implementing a custom loss function in PyTorch for highly imbalanced multi-class image segmentation.",
    "Troubleshooting 502 Bad Gateway errors in Kubernetes NGINX Ingress Controller behind an AWS ALB.",
    "Calculating the aerodynamic drag coefficient using OpenFOAM for a simplified Formula SAE car.",
    "Resolving dependency conflicts between React 18 concurrent mode and legacy Redux middleware.",
    "Using WebGL shaders to implement physically based rendering (PBR) materials.",
    "Designing a microstrip patch antenna for 2.4 GHz using HFSS.",
    "Mitigating SSRF vulnerabilities in a Node.js microservice handling user-provided webhooks.",
    "Configuring a reverse proxy to handle WebSocket connections with mutual TLS authentication.",
    # Task-Oriented / Generative
    "Write a python script to scrape product titles and prices from a Shopify store.",
    "Draft a cold outreach LinkedIn message for a SaaS product targeting HR directors.",
    "Generate a 30-day social media content calendar for a local coffee shop.",
    "Create a standard operating procedure (SOP) for onboarding new freelance writers.",
    "Write a comprehensive prompt to get an AI to act as a rigorous technical interviewer.",
    "Develop a workout plan for a beginner training for a 5k with bad knees.",
    "Outline a five-slide pitch deck for a pre-seed AI health-tech startup.",
    "Write a bash script that backs up a PostgreSQL database and uploads it to an S3 bucket daily.",
    "Create a template for a weekly one-on-one meeting agenda between a manager and a direct report.",
    "Generate a list of 20 creative name ideas for a plant-based dog food brand.",
]

print(f"Queries: {len(test_queries)}")


# Nomic  →  nomic-ai/nomic-embed-text-v1.5  (Matryoshka 768d, truncate)

print("\n[1/3] nomic-ai/nomic-embed-text-v1.5  (256 / 512 / 768d)")
MODEL_NOMIC = "nomic-ai/nomic-embed-text-v1.5"

# nomic-embed requires a task prefix for retrieval queries
prefixed = [f"search_query: {q}" for q in test_queries]

model_nomic = SentenceTransformer(MODEL_NOMIC, trust_remote_code=True)

# Encode at full 768d, then slice for smaller dims
emb_768 = model_nomic.encode(
    prefixed,
    convert_to_numpy=True,
    show_progress_bar=True,
    normalize_embeddings=False,   # unit-norm before slicing = correct Matryoshka usage
)
print(f"  Full shape: {emb_768.shape}")

for dim, fname in [(256, "queries_emb_nomic_256d.npy"),
                   (512, "queries_emb_nomic_512d.npy"),
                   (768, "queries_emb_nomic_768d.npy")]:
    sliced = emb_768[:, :dim].copy()
    # re-normalise after slicing (standard Matryoshka practice)
    norms = np.linalg.norm(sliced, axis=1, keepdims=True)
    sliced = sliced / (norms + 1e-12)
    out = OUT_NOMIC / fname
    np.save(out, sliced.astype(np.float32))
    print(f"  Saved {dim}d → {out}  shape={sliced.shape}")

del model_nomic  # free GPU / RAM


# 2.  all-MiniLM  →  sentence-transformers/all-MiniLM-L6-v2  (384d)
#     covers: emb_02, emb_05

print("\n[2/3] sentence-transformers/all-MiniLM-L6-v2  (384d)  →  emb_02 / emb_05")
MODEL_384 = "sentence-transformers/all-MiniLM-L6-v2"

model_384 = SentenceTransformer(MODEL_384)
emb_384 = model_384.encode(
    test_queries,
    convert_to_numpy=True,
    show_progress_bar=True,
    normalize_embeddings=False,
)
print(f"  Shape: {emb_384.shape}")

out_384 = OUT_STD / "queries_emb_384.npy"
np.save(out_384, emb_384.astype(np.float32))
print(f"  Saved → {out_384}")

del model_384


MODEL_1024 = 'perplexity-ai/pplx-embed-v1-0.6b'
model_1024 = SentenceTransformer(MODEL_1024, trust_remote_code=True)  
queries_1024 = test_queries

emb_1024 = model_1024.encode(
    queries_1024,
    convert_to_numpy=True,
    show_progress_bar=True,
    normalize_embeddings=False,
)
print(f"  Shape: {emb_1024.shape}")

out_1024 = OUT_STD / "queries_emb_1024.npy"
np.save(out_1024, emb_1024.astype(np.float32))
print(f"  Saved → {out_1024}")

del model_1024


for p in sorted(list(OUT_NOMIC.glob("*.npy")) + list(OUT_STD.glob("*.npy"))):
    arr = np.load(p)
    print(f"  {p.relative_to(ROOT)}  {arr.shape}")
print("="*60)
