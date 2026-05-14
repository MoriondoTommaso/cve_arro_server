"""
prompt_benchmark_nomic768.py

Full metric evaluation on the PromptKaban corpus using the winning
embedding pair: nomic-embed-text-v1.5 @ 768d.

Runs the EXACT same metric suite as test_17_CVE_neurips_v2.py:
  - Spearman ρ / Kendall τ ranking agreement
  - NDCG@25 (Hybrid and Taumode vs Cosine as internal reference)
  - Head-tail quality metrics (T/H ratio, tail CV, tail decay)
  - Traditional / Semantic / Tolerant Recall (Kuffo et al. proxy)
  - HEAD_K sensitivity sweep over [3, 5, 10]

Outputs mirror the CVE script outputs (renamed with prompt_ prefix):
  - prompt_search_results.csv
  - prompt_comparison_metrics.csv
  - prompt_tail_metrics.csv
  - prompt_semantic_recall_metrics.csv
  - prompt_summary.csv
  - prompt_query_comparison.txt
  - prompt_run_metadata.json
  - prompt_headk_sweep.csv
  - prompt_top25_comparison.png
  - prompt_tail_analysis.png
  - prompt_semantic_recall_comparison.png
  - prompt_metric_deltas.png
  - prompt_win_loss_heatmap.png
  - prompt_pareto_tradeoff.png
  - prompt_headk_sweep.png

Usage:
    python prompt_benchmark_nomic768.py --corpus /path/to/prompts.jsonl [--corpus-emb /path/to/nomic_embs_768.npy]

The --corpus argument accepts:
  - A .jsonl / .json file where each line is {"id": ..., "title": ..., "text": ...}
  - A plain .txt file  (one prompt per line; title = first 80 chars)
  - A directory of .txt files

If --corpus-emb is given the corpus embeddings are loaded from disk instead of
being recomputed (saves time on re-runs).

Notes:
The test_queries are hardcoded in the script for simplicity, but can be easily modified
reference:
https://github.com/tuned-org-uk/pyarrowspace/blob/main/neurips/CVE/test_17_CVE_neurips_v2.py
"""

import argparse
import csv
import json
import logging
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


# Configuration  (identical to CVE script defaults)
TAU_COSINE   = 1.0
TAU_HYBRID   = 0.72
TAU_TAUMODE  = 0.42

NDCG_K       = 25
RESULTS_K    = 25
NEIGHBOUR_K  = 30
HEAD_K       = 3
HEAD_K_SWEEP = [3, 5, 10]

NOMIC_MODEL  = "nomic-ai/nomic-embed-text-v1.5"
NOMIC_DIM    = 768         

OUTPUT_DIR   = Path("output_prompt_nomic768")

TAU_METHOD_KEYS = ["Cosine", "Hybrid", "Taumode"]
TAU_DISPLAY = {
    "Cosine":  f"Cosine (τ={TAU_COSINE})",
    "Hybrid":  f"hybrid (τ={TAU_HYBRID})",
    "Taumode": f"taumode (τ={TAU_TAUMODE})",
}
TAU_VALUES = {"Cosine": TAU_COSINE, "Hybrid": TAU_HYBRID, "Taumode": TAU_TAUMODE}
METHOD_COLORS = {"Cosine": "#1f77b4", "Hybrid": "#ff7f0e", "Taumode": "#2ca02c"}

# graph_params reuse the same defaults as CVE benchmark
graph_params = {
    "eps":   1.31,
    "k":     NEIGHBOUR_K,
    "topk":  RESULTS_K,
    "p":     1.8,
    "sigma": 0.535,
}


# 40-query evaluation set


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

# Data Loading
def load_corpus(corpus_path: str):
    """
    Load prompt corpus from:
      - .jsonl  → each line: {"id": ..., "title": ..., "text": ...}
      - .json   → list of objects with same keys
      - .txt    → one prompt per line, title = first 80 chars
      - dir     → each .txt file is one document
    Returns: ids[], titles[], docs[]
    """
    p = Path(corpus_path)
    ids, titles, docs = [], [], []

    if p.is_dir():
        for f in sorted(p.glob("*.txt")):
            text = f.read_text(encoding="utf-8").strip()
            ids.append(f.stem)
            titles.append(text[:80])
            docs.append(text)

    elif p.suffix in (".jsonl",):
        with open(p, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ids.append(str(obj.get("id", i)))
                titles.append(str(obj.get("title", obj.get("text", "")[:80])))
                docs.append(str(obj.get("text", obj.get("content", ""))))

    elif p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for i, obj in enumerate(data):
                ids.append(str(obj.get("id", i)))
                titles.append(str(obj.get("title", obj.get("text", "")[:80])))
                docs.append(str(obj.get("text", obj.get("content", ""))))
        else:
            raise ValueError("JSON corpus must be a list of objects.")

    elif p.suffix == ".txt":
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if line:
                ids.append(str(i))
                titles.append(line[:80])
                docs.append(line)

    else:
        raise ValueError(f"Unsupported corpus format: {p.suffix}")

    if not docs:
        raise SystemExit(f"No documents found in {corpus_path}")
    print(f"Loaded {len(docs)} corpus documents.")
    return ids, titles, docs


def build_embeddings(texts, model_name=NOMIC_MODEL, dim=NOMIC_DIM, cache_file=None):
    """
    Encode texts with nomic-embed-text-v1.5 at the requested matryoshka dim.
    Loads from cache_file if it exists and matches the corpus size.
    Applies the same ×12 scale factor as the CVE script.
    """
    if cache_file and Path(cache_file).exists():
        print(f"Loading cached embeddings from {cache_file}...")
        X = np.load(cache_file)
        if len(X) == len(texts):
            print(f"Embeddings loaded. Shape: {X.shape}")
            return X
        print(f"Cache size mismatch ({len(X)} vs {len(texts)}). Regenerating.")

    print(f"Loading model {model_name} ...")
    model = SentenceTransformer(model_name, trust_remote_code=True)

    print(f"Encoding {len(texts)} texts at dim={dim}...")
    X = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    # Truncate to requested matryoshka dimension
    X = X[:, :dim].astype(np.float64)
    X_scaled = X * 1.2e1   # same scale factor as CVE script

    if cache_file:
        np.save(cache_file, X_scaled)
        print(f"Embeddings cached to {cache_file}")

    print(f"Embeddings shape: {X_scaled.shape}, sample: {X_scaled[0][:5]}...")
    return X_scaled


# Core Metrics  (verbatim from test_17_CVE_neurips_v2.py)
def compute_ranking_metrics(results_a, results_b):
    indices_a = [idx for idx, _ in results_a]
    indices_b = [idx for idx, _ in results_b]
    shared = set(indices_a) & set(indices_b)
    if len(shared) < 2:
        return 0.0, 0.0
    rank_a = [indices_a.index(idx) for idx in shared]
    rank_b = [indices_b.index(idx) for idx in shared]
    rho, _ = spearmanr(rank_a, rank_b)
    tau, _ = kendalltau(rank_a, rank_b)
    return (0.0 if np.isnan(rho) else float(rho),
            0.0 if np.isnan(tau) else float(tau))


def compute_ndcg(results_pred, results_ref, k=NDCG_K):
    ref_indices = [idx for idx, _ in results_ref[:k]]
    relevance_map = {idx: k - i for i, idx in enumerate(ref_indices)}
    pred_indices = [idx for idx, _ in results_pred[:k]]
    true_relevance = [relevance_map.get(idx, 0) for idx in pred_indices]
    if sum(true_relevance) == 0:
        return 0.0
    try:
        pred_scores = np.array([score for _, score in results_pred[:k]])
        if pred_scores.max() > 0:
            pred_scores = pred_scores / pred_scores.max()
        return float(ndcg_score(
            np.array([true_relevance]).reshape(1, -1),
            np.array([pred_scores]).reshape(1, -1),
            k=k,
        ))
    except Exception:
        return 0.0


def analyze_tail_distribution(results_list, labels, k_head=3, k_tail=20):
    min_length = min(len(r) for r in results_list)
    if min_length <= k_head:
        return {}
    actual_k_tail = min(k_tail, min_length)
    metrics = {}
    for results, label in zip(results_list, labels):
        seg = results[:actual_k_tail]
        head_scores = [s for _, s in seg[:k_head]]
        tail_scores = [s for _, s in seg[k_head:actual_k_tail]]
        if not tail_scores or not head_scores:
            continue
        tail_mean = float(np.mean(tail_scores))
        tail_std  = float(np.std(tail_scores))
        head_mean = float(np.mean(head_scores))
        th_ratio  = tail_mean / head_mean if head_mean > 1e-10 else 0.0
        tail_cv   = tail_std / tail_mean   if tail_mean > 1e-10 else 0.0
        tail_decay = ((tail_scores[0] - tail_scores[-1]) / len(tail_scores)
                      if len(tail_scores) > 1 else 0.0)
        metrics[label] = {
            "head_mean": head_mean, "tail_mean": tail_mean, "tail_std": tail_std,
            "tail_to_head_ratio": float(th_ratio), "tail_cv": float(tail_cv),
            "tail_decay_rate": float(tail_decay),
            "n_tail_items": len(tail_scores), "total_items": actual_k_tail,
        }
    return metrics

#Semantic Recall (Kuffo et al. proxy) 

def compute_traditional_recall(retrieved_ids, ground_truth_ids):
    if not ground_truth_ids:
        return 0.0
    return len(set(retrieved_ids) & set(ground_truth_ids)) / len(ground_truth_ids)


def identify_semantic_neighbors(gt_ids, gt_scores, score_gap_percentile=25.0):
    if not gt_scores:
        return []
    scores = np.array(gt_scores)
    threshold = np.percentile(scores, 100 - score_gap_percentile)
    return [i for i, s in zip(gt_ids, gt_scores) if s >= threshold]


def compute_semantic_recall(retrieved_ids, gt_ids, sn_ids):
    sn_set = set(sn_ids) & set(gt_ids)
    if not sn_set:
        return float("nan")
    return len(set(retrieved_ids) & sn_set) / len(sn_set)


def estimate_tolerance_threshold(gt_scores, k):
    scores = list(gt_scores)[:k]
    if len(scores) < 2:
        return 1.0
    max_score = max(scores) if max(scores) > 0 else 1.0
    two_thirds_k = max(0, int(2 * k / 3) - 1)
    return max(0.1, abs(scores[two_thirds_k] - scores[-1]) / max_score * 100.0)


def compute_tolerant_recall(retrieved_ids, retrieved_scores,
                             gt_ids, gt_scores, tolerance_pct=1.0):
    if not gt_ids:
        return 0.0
    k = len(gt_ids)
    gt_map = {i: s for i, s in zip(gt_ids, gt_scores)}
    matched, cnt = set(), 0
    for ri, rs in zip(retrieved_ids, retrieved_scores):
        if ri in gt_map and ri not in matched:
            matched.add(ri); cnt += 1
        else:
            for gi, gs in zip(gt_ids, gt_scores):
                if gi in matched:
                    continue
                if rs >= gs * (1.0 - tolerance_pct / 100.0):
                    matched.add(gi); cnt += 1; break
    return cnt / k


def compute_all_recall_metrics(retrieved_ids, retrieved_scores,
                                gt_ids, gt_scores,
                                tolerance_pct=None, sn_pct=25.0):
    k = len(gt_ids)
    trad = compute_traditional_recall(retrieved_ids, gt_ids)
    sn_ids = identify_semantic_neighbors(gt_ids, gt_scores, sn_pct)
    sem  = compute_semantic_recall(retrieved_ids, gt_ids, sn_ids)
    if tolerance_pct is None:
        tolerance_pct = estimate_tolerance_threshold(gt_scores, k)
    tol = compute_tolerant_recall(retrieved_ids, retrieved_scores,
                                   gt_ids, gt_scores, tolerance_pct)
    return {
        "traditional_recall": float(trad),
        "semantic_recall":    sem,
        "tolerant_recall":    float(tol),
        "n_semantic_neighbors": int(len(sn_ids)),
        "tolerance_pct_used": float(tolerance_pct),
    }


# ArrowSpace tau search  (pure-numpy fallback identical to the notebook)
def _cosine_scores(C, q):
    cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    qn = q / (np.linalg.norm(q) + 1e-12)
    return cn @ qn


def _topk(scores, k):
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx.tolist(), scores[idx].tolist()


def tau_search(C, q, tau, k=RESULTS_K):
    """
    Mirrors _tau_search() from the notebook and the ArrowSpace search logic:
      tau=1.0  → pure cosine
      tau<1.0  → blend cosine + Rayleigh energy (Hybrid / Taumode)
    """
    sc = _cosine_scores(C, q)
    if tau < 1.0:
        idx, _ = _topk(sc, min(k * 4, len(C)))
        sel_C = C[idx]
        cn = sel_C / (np.linalg.norm(sel_C, axis=1, keepdims=True) + 1e-12)
        qn = q / (np.linalg.norm(q) + 1e-12)
        cos_sc = cn @ qn
        energy = np.array([
            np.dot(cn[i], qn) ** 2 / (np.dot(cn[i], cn[i]) + 1e-12)
            for i in range(len(cn))
        ])
        blended = tau * cos_sc + (1 - tau) * energy
        order = np.argsort(blended)[::-1][:k]
        return [(idx[j], float(blended[j])) for j in order]
    else:
        idx, scores = _topk(sc, min(k, len(C)))
        return list(zip(idx, scores))


# CSV / JSON export helpers  (same structure as CVE script)
def save_search_results(queries, all_results, ids, titles, out):
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query_id","query_text","tau_method",
                                           "rank","doc_id","title","score"])
        w.writeheader()
        for qi, query in enumerate(queries):
            for tau_key, results in zip(TAU_METHOD_KEYS, all_results[qi]):
                for rank, (idx, score) in enumerate(results[:RESULTS_K], 1):
                    w.writerow({"query_id": qi+1, "query_text": query,
                                "tau_method": tau_key, "rank": rank,
                                "doc_id": ids[idx], "title": titles[idx],
                                "score": f"{score:.6f}"})
    print(f"Search results → {out}")


def save_comparison_metrics(comparison_metrics, out):
    fields = ["query_id","query_text","min_length",
              "spearman_cosine_hybrid","spearman_cosine_taumode","spearman_hybrid_taumode",
              "kendall_cosine_hybrid","kendall_cosine_taumode","kendall_hybrid_taumode",
              "ndcg_hybrid_vs_cosine","ndcg_taumode_vs_cosine","ndcg_taumode_vs_hybrid"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for qi, m in enumerate(comparison_metrics):
            w.writerow({"query_id": qi+1, "query_text": m["query"],
                        "min_length": m["min_length"],
                        "spearman_cosine_hybrid":  f"{m['spearman'][0]:.6f}",
                        "spearman_cosine_taumode": f"{m['spearman'][1]:.6f}",
                        "spearman_hybrid_taumode": f"{m['spearman'][2]:.6f}",
                        "kendall_cosine_hybrid":   f"{m['kendall'][0]:.6f}",
                        "kendall_cosine_taumode":  f"{m['kendall'][1]:.6f}",
                        "kendall_hybrid_taumode":  f"{m['kendall'][2]:.6f}",
                        "ndcg_hybrid_vs_cosine":   f"{m['ndcg'][0]:.6f}",
                        "ndcg_taumode_vs_cosine":  f"{m['ndcg'][1]:.6f}",
                        "ndcg_taumode_vs_hybrid":  f"{m['ndcg'][2]:.6f}"})
    print(f"Comparison metrics → {out}")


def save_tail_metrics(comparison_metrics, out):
    fields = ["query_id","query_text","tau_method","head_mean","tail_mean",
              "tail_std","tail_to_head_ratio","tail_cv","tail_decay_rate",
              "n_tail_items","total_items"]
    tau_labels = [TAU_DISPLAY[k] for k in TAU_METHOD_KEYS]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for qi, m in enumerate(comparison_metrics):
            for lbl in tau_labels:
                if lbl not in m.get("tail_metrics", {}): continue
                tm = m["tail_metrics"][lbl]
                w.writerow({"query_id": qi+1, "query_text": m["query"],
                            "tau_method": lbl,
                            "head_mean":          f"{tm['head_mean']:.6f}",
                            "tail_mean":          f"{tm['tail_mean']:.6f}",
                            "tail_std":           f"{tm['tail_std']:.6f}",
                            "tail_to_head_ratio": f"{tm['tail_to_head_ratio']:.6f}",
                            "tail_cv":            f"{tm['tail_cv']:.6f}",
                            "tail_decay_rate":    f"{tm['tail_decay_rate']:.6f}",
                            "n_tail_items":       tm["n_tail_items"],
                            "total_items":        tm["total_items"]})
    print(f"Tail metrics → {out}")


def save_semantic_recall(comparison_metrics, out):
    fields = ["query_id","query_text","tau_method","traditional_recall",
              "semantic_recall","tolerant_recall","n_semantic_neighbors",
              "tolerance_pct_used","tolerant_minus_traditional"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for qi, m in enumerate(comparison_metrics):
            for tau_key in TAU_METHOD_KEYS:
                rm = m.get("recall_metrics", {}).get(tau_key)
                if not rm: continue
                sem = rm["semantic_recall"]
                is_nan = isinstance(sem, float) and math.isnan(sem)
                w.writerow({"query_id": qi+1, "query_text": m["query"],
                            "tau_method": tau_key,
                            "traditional_recall": f"{rm['traditional_recall']:.6f}",
                            "semantic_recall":    "nan" if is_nan else f"{sem:.6f}",
                            "tolerant_recall":    f"{rm['tolerant_recall']:.6f}",
                            "n_semantic_neighbors": rm["n_semantic_neighbors"],
                            "tolerance_pct_used": f"{rm['tolerance_pct_used']:.4f}",
                            "tolerant_minus_traditional": f"{rm['tolerant_recall']-rm['traditional_recall']:.6f}"})
    print(f"Semantic recall → {out}")


def save_summary(comparison_metrics, out):
    fields = ["metric_type","metric_name","value","std_dev"]
    tau_labels = [TAU_DISPLAY[k] for k in TAU_METHOD_KEYS]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for pair, col in [("hybrid vs Cosine",0),("taumode vs Cosine",1),("taumode vs hybrid",2)]:
            vals = [m["ndcg"][col] for m in comparison_metrics]
            w.writerow({"metric_type": f"NDCG@{NDCG_K}", "metric_name": pair,
                        "value": f"{np.mean(vals):.6f}", "std_dev": f"{np.std(vals):.6f}"})
        valid_tail = [m for m in comparison_metrics if m["tail_metrics"]]
        for lbl in tau_labels:
            ratios = [m["tail_metrics"][lbl]["tail_to_head_ratio"]
                      for m in valid_tail if lbl in m["tail_metrics"]]
            if ratios:
                w.writerow({"metric_type": "Tail/Head Ratio", "metric_name": lbl,
                            "value": f"{np.mean(ratios):.6f}", "std_dev": f"{np.std(ratios):.6f}"})
        for tau_key in TAU_METHOD_KEYS:
            for metric_type, key in [("Traditional Recall@k","traditional_recall"),
                                      ("Semantic Recall@k","semantic_recall"),
                                      ("Tolerant Recall@k","tolerant_recall")]:
                vals = []
                for m in comparison_metrics:
                    rm = m.get("recall_metrics",{}).get(tau_key)
                    if rm:
                        v = rm[key]
                        if not (isinstance(v, float) and math.isnan(v)):
                            vals.append(v)
                if vals:
                    w.writerow({"metric_type": metric_type, "metric_name": tau_key,
                                "value": f"{np.mean(vals):.6f}", "std_dev": f"{np.std(vals):.6f}"})
    print(f"Summary → {out}")


def save_headk_sweep(rows, out):
    if not rows: return
    fields = ["head_k","query_id","query_text","tau_method","head_mean","tail_mean",
              "tail_std","tail_to_head_ratio","tail_cv","tail_decay_rate",
              "n_tail_items","total_items"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in rows: w.writerow(row)
    print(f"HEAD_K sweep → {out}")


def run_headk_sweep(queries, all_results, head_k_values):
    rows = []
    tau_labels = [TAU_DISPLAY[k] for k in TAU_METHOD_KEYS]
    for h in head_k_values:
        for qi, query in enumerate(queries):
            rc, rh, rt = all_results[qi]
            min_len = min(len(rc), len(rh), len(rt))
            if min_len <= h or (min_len - h) < 2:
                continue
            tail_m = analyze_tail_distribution(
                [rc[:min_len], rh[:min_len], rt[:min_len]],
                tau_labels, k_head=h, k_tail=min_len)
            for lbl in tau_labels:
                if lbl not in tail_m: continue
                m = tail_m[lbl]
                rows.append({"head_k": h, "query_id": qi+1, "query_text": query,
                             "tau_method": lbl,
                             "head_mean":          f"{m['head_mean']:.6f}",
                             "tail_mean":          f"{m['tail_mean']:.6f}",
                             "tail_std":           f"{m['tail_std']:.6f}",
                             "tail_to_head_ratio": f"{m['tail_to_head_ratio']:.6f}",
                             "tail_cv":            f"{m['tail_cv']:.6f}",
                             "tail_decay_rate":    f"{m['tail_decay_rate']:.6f}",
                             "n_tail_items":       m["n_tail_items"],
                             "total_items":        m["total_items"]})
    return rows


def save_run_metadata(out, corpus_path, n_docs, n_queries, comparison_metrics):
    min_lens = [m["min_length"] for m in comparison_metrics] if comparison_metrics else []
    payload = {
        "test_name":          "prompt_benchmark_nomic768",
        "timestamp_unix":     time.time(),
        "corpus_path":        str(corpus_path),
        "n_documents":        int(n_docs),
        "embedding_model":    NOMIC_MODEL,
        "embedding_dim":      NOMIC_DIM,
        "tau_values":         TAU_VALUES,
        "head_k":             HEAD_K,
        "head_k_sweep":       HEAD_K_SWEEP,
        "neighbour_k":        NEIGHBOUR_K,
        "graph_params":       graph_params,
        "query_count":        int(n_queries),
        "ndcg_k":             NDCG_K,
        "results_k":          RESULTS_K,
        "min_result_length":  int(min(min_lens)) if min_lens else None,
        "max_result_length":  int(max(min_lens)) if min_lens else None,
        "note": (
            "Cosine ranking is the internal reference G for recall comparisons. "
            "This mirrors the methodological note in test_17_CVE_neurips_v2.py."
        ),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Run metadata → {out}")


# Plots  (same panel layout as CVE script)
def plot_top25(queries, all_results, ids, titles, out):
    n = len(queries)
    fig, axes = plt.subplots(n, 3, figsize=(18, 6*n))
    if n == 1: axes = axes.reshape(1, -1)
    tau_labels = [TAU_DISPLAY[k] for k in TAU_METHOD_KEYS]
    colors = [METHOD_COLORS[k] for k in TAU_METHOD_KEYS]
    for qi, query in enumerate(queries):
        k = min(RESULTS_K, min(len(r) for r in all_results[qi]))
        for ti, (results, lbl, color) in enumerate(zip(all_results[qi], tau_labels, colors)):
            ax = axes[qi, ti]
            scores = [s for _, s in results[:k]]
            ax.bar(range(1, k+1), scores, alpha=0.75, color=color)
            ax.set_title(f"Q{qi+1}: {lbl}\n{query[:50]}...", fontsize=9, fontweight="bold")
            ax.set_xlabel("Rank"); ax.set_ylabel("Score"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Top-25 plot → {out}")


def plot_tail_analysis(queries, all_results, out):
    n = len(queries)
    tau_labels = [TAU_DISPLAY[k] for k in TAU_METHOD_KEYS]
    colors = [METHOD_COLORS[k] for k in TAU_METHOD_KEYS]
    fig = plt.figure(figsize=(20, 5*n))
    gs = fig.add_gridspec(n, 4, hspace=0.3, wspace=0.3)
    for qi, query in enumerate(queries):
        rc, rh, rt = all_results[qi]
        k = min(len(rc), len(rh), len(rt))
        trimmed = [rc[:k], rh[:k], rt[:k]]

        ax1 = fig.add_subplot(gs[qi, 0])
        for res, lbl, col in zip(trimmed, tau_labels, colors):
            ax1.plot(range(1,k+1), [s for _,s in res], marker="o",
                     label=lbl, color=col, alpha=0.75, markersize=4, linewidth=2)
        ax1.axvline(x=HEAD_K+0.5, color="red", ls="--", alpha=0.5)
        ax1.set_title(f"Q{qi+1} Score dist\n{query[:45]}...", fontsize=9, fontweight="bold")
        ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

        ax2 = fig.add_subplot(gs[qi, 1])
        if k > HEAD_K:
            for res, lbl, col in zip(trimmed, tau_labels, colors):
                ax2.plot(range(HEAD_K+1, k+1), [s for _,s in res[HEAD_K:]],
                         marker="s", label=lbl, color=col, alpha=0.75)
            ax2.set_title(f"Q{qi+1} Tail ranks", fontsize=9, fontweight="bold")
            ax2.grid(alpha=0.3)

        ax3 = fig.add_subplot(gs[qi, 2])
        if k > HEAD_K:
            bp = ax3.boxplot([[s for _,s in r[HEAD_K:]] for r in trimmed],
                             labels=["Cos","Hyb","Tau"], patch_artist=True, widths=0.6)
            for patch, col in zip(bp["boxes"], colors):
                patch.set_facecolor(col); patch.set_alpha(0.6)
            ax3.set_title(f"Q{qi+1} Tail variability", fontsize=9, fontweight="bold")
            ax3.grid(axis="y", alpha=0.3)

        ax4 = fig.add_subplot(gs[qi, 3])
        if k > HEAD_K:
            tm = analyze_tail_distribution(trimmed, tau_labels, k_head=HEAD_K, k_tail=k)
            x = np.arange(3); w = 0.25
            for i, (lbl, col) in enumerate(zip(tau_labels, colors)):
                if lbl in tm:
                    m = tm[lbl]
                    vals = [m["tail_mean"], m["tail_to_head_ratio"],
                            1.0/(1.0+m["tail_cv"]) if m["tail_cv"] > 0 else 1.0]
                    ax4.bar(x + i*w, vals, w, label=lbl, color=col, alpha=0.75)
            ax4.set_xticks(x+w); ax4.set_xticklabels(["T-Mean","T/H","Stab"], fontsize=9)
            ax4.set_title(f"Q{qi+1} Tail metrics", fontsize=9, fontweight="bold")
            ax4.legend(fontsize=7); ax4.grid(axis="y", alpha=0.3)

    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Tail analysis → {out}")


def plot_semantic_recall(comparison_metrics, out):
    tau_methods = TAU_METHOD_KEYS
    n = len(comparison_metrics)
    fig, axes = plt.subplots(len(tau_methods), 3, figsize=(20, 6*len(tau_methods)))
    fig.suptitle("Semantic Recall — Traditional vs Semantic vs Tolerant\n"
                 "(proxy inspired by Kuffo et al., SIGIR '26)",
                 fontsize=13, fontweight="bold", y=1.01)
    mc = {"traditional": "#4c72b0", "semantic": "#55a868", "tolerant": "#dd8452"}
    for ti, tau_key in enumerate(tau_methods):
        trad_v, sem_v, tol_v, sn_v, vmask = [], [], [], [], []
        for m in comparison_metrics:
            rm = m.get("recall_metrics",{}).get(tau_key)
            if rm:
                trad_v.append(rm["traditional_recall"])
                sr = rm["semantic_recall"]
                is_nan = isinstance(sr, float) and math.isnan(sr)
                sem_v.append(0.0 if is_nan else sr)
                tol_v.append(rm["tolerant_recall"])
                sn_v.append(rm["n_semantic_neighbors"])
                vmask.append(not is_nan)
            else:
                trad_v.append(0.0); sem_v.append(0.0); tol_v.append(0.0)
                sn_v.append(0); vmask.append(False)
        x = np.arange(n); bw = 0.28
        ax0 = axes[ti, 0]
        ax0.bar(x-bw, trad_v, bw, label="Traditional", color=mc["traditional"], alpha=0.85)
        ax0.bar(x,    sem_v,  bw, label="Semantic",    color=mc["semantic"],    alpha=0.85)
        ax0.bar(x+bw, tol_v,  bw, label="Tolerant",    color=mc["tolerant"],    alpha=0.85)
        ax0.set_title(f"{TAU_DISPLAY[tau_key]}\nRecall per Query", fontweight="bold")
        ax0.set_xticks(x); ax0.set_xticklabels([f"Q{i+1}" for i in range(n)], rotation=45, ha="right", fontsize=7)
        ax0.set_ylim(0, 1.15); ax0.legend(fontsize=8); ax0.grid(axis="y", alpha=0.3)
        ax1 = axes[ti, 1]
        vt = [v for v, ok in zip(trad_v, vmask) if ok]
        vs = [v for v, ok in zip(sem_v,  vmask) if ok]
        vsn= [v for v, ok in zip(sn_v,   vmask) if ok]
        if vt:
            sc = ax1.scatter(vt, vs, c=vsn, cmap="viridis", s=60, alpha=0.8)
            plt.colorbar(sc, ax=ax1, label="#SN", shrink=0.8)
            ax1.plot([0,1],[0,1],"r--", alpha=0.7)
        ax1.set_title(f"{TAU_DISPLAY[tau_key]}\nTrad vs Semantic", fontweight="bold")
        ax1.set_xlim(-0.05,1.1); ax1.set_ylim(-0.05,1.1); ax1.grid(alpha=0.3)
        ax2 = axes[ti, 2]
        uplift = [t-tr for t,tr in zip(tol_v, trad_v)]
        if uplift:
            ax2.hist(uplift, bins=min(15,len(uplift)), color=mc["tolerant"], alpha=0.8, edgecolor="white")
            ax2.axvline(0, color="red", lw=1.5, ls="--")
            ax2.axvline(np.mean(uplift), color="orange", lw=1.5,
                        label=f"Mean: {np.mean(uplift):+.3f}")
            ax2.set_title(f"{TAU_DISPLAY[tau_key]}\nTolerant Uplift Dist.", fontweight="bold")
            ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Semantic recall plot → {out}")


def plot_metric_deltas(comparison_metrics, out):
    n = len(comparison_metrics)
    labels = [f"Q{i+1}" for i in range(n)]
    hyb_th, tau_th, hyb_sem, tau_sem, hyb_tol, tau_tol = [], [], [], [], [], []
    for m in comparison_metrics:
        tm = m["tail_metrics"]
        cl = TAU_DISPLAY["Cosine"]; hl = TAU_DISPLAY["Hybrid"]; tl = TAU_DISPLAY["Taumode"]
        hyb_th.append(tm[hl]["tail_to_head_ratio"] - tm[cl]["tail_to_head_ratio"])
        tau_th.append(tm[tl]["tail_to_head_ratio"] - tm[cl]["tail_to_head_ratio"])
        rm = m["recall_metrics"]
        cs = rm["Cosine"]["semantic_recall"]; hs = rm["Hybrid"]["semantic_recall"]; ts = rm["Taumode"]["semantic_recall"]
        hyb_sem.append((hs-cs) if not any(isinstance(v,float) and math.isnan(v) for v in [hs,cs]) else 0.0)
        tau_sem.append((ts-cs) if not any(isinstance(v,float) and math.isnan(v) for v in [ts,cs]) else 0.0)
        hyb_tol.append(rm["Hybrid"]["tolerant_recall"] - rm["Cosine"]["tolerant_recall"])
        tau_tol.append(rm["Taumode"]["tolerant_recall"] - rm["Cosine"]["tolerant_recall"])
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    x = np.arange(n); w = 0.36
    for ax, (title, hv, tv) in zip(axes, [
        ("Δ Tail/Head Ratio vs Cosine",   hyb_th,  tau_th),
        ("Δ Semantic Recall vs Cosine",   hyb_sem, tau_sem),
        ("Δ Tolerant Recall vs Cosine",   hyb_tol, tau_tol),
    ]):
        ax.bar(x-w/2, hv, w, color=METHOD_COLORS["Hybrid"],  alpha=0.8, label="hybrid − Cosine")
        ax.bar(x+w/2, tv, w, color=METHOD_COLORS["Taumode"], alpha=0.8, label="taumode − Cosine")
        ax.axhline(0, color="black", lw=1); ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    axes[-1].set_xticks(x); axes[-1].set_xticklabels(labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Metric deltas → {out}")


def plot_win_loss_heatmap(comparison_metrics, out):
    metric_names = ["T/H Ratio","Tail CV","Tail Decay","Semantic Recall","Tolerant Recall"]
    n = len(comparison_metrics)
    win_matrix = np.zeros((len(metric_names), n))
    annotations = [["" for _ in range(n)] for _ in range(len(metric_names))]
    m2v = {"Cosine": 0, "Hybrid": 1, "Taumode": 2}
    higher = {"T/H Ratio", "Semantic Recall", "Tolerant Recall"}
    for qi, m in enumerate(comparison_metrics):
        tm = m["tail_metrics"]; rm = m["recall_metrics"]
        vals = {
            "T/H Ratio":      {k: tm[TAU_DISPLAY[k]]["tail_to_head_ratio"] for k in TAU_METHOD_KEYS},
            "Tail CV":        {k: tm[TAU_DISPLAY[k]]["tail_cv"]            for k in TAU_METHOD_KEYS},
            "Tail Decay":     {k: tm[TAU_DISPLAY[k]]["tail_decay_rate"]    for k in TAU_METHOD_KEYS},
            "Semantic Recall":{k: (-1 if (isinstance(rm[k]["semantic_recall"],float) and math.isnan(rm[k]["semantic_recall"])) else rm[k]["semantic_recall"]) for k in TAU_METHOD_KEYS},
            "Tolerant Recall":{k: rm[k]["tolerant_recall"] for k in TAU_METHOD_KEYS},
        }
        for mi, metric in enumerate(metric_names):
            mv = vals[metric]
            winner = max(mv, key=mv.get) if metric in higher else min(mv, key=mv.get)
            win_matrix[mi, qi] = m2v[winner]
            annotations[mi][qi] = winner[0]
    cmap = plt.matplotlib.colors.ListedColormap([METHOD_COLORS[k] for k in TAU_METHOD_KEYS])
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.imshow(win_matrix, aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5)
    ax.set_xticks(np.arange(n)); ax.set_xticklabels([f"Q{i+1}" for i in range(n)], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(metric_names))); ax.set_yticklabels(metric_names)
    ax.set_title("Per-query metric winners (C=Cosine, H=hybrid, T=taumode)", fontweight="bold")
    for i in range(len(metric_names)):
        for j in range(n):
            ax.text(j, i, annotations[i][j], ha="center", va="center", color="white", fontweight="bold")
    ax.legend(handles=[
        plt.Line2D([0],[0], marker="s", color="w", markerfacecolor=METHOD_COLORS[k], markersize=12, label=k)
        for k in TAU_METHOD_KEYS
    ], loc="upper center", bbox_to_anchor=(0.5,-0.12), ncol=3)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Win/loss heatmap → {out}")


def plot_pareto(comparison_metrics, out):
    fig, ax = plt.subplots(figsize=(10, 8))
    for method in TAU_METHOD_KEYS:
        xs = [m["tail_metrics"][TAU_DISPLAY[method]]["tail_to_head_ratio"] for m in comparison_metrics]
        ys = [m["recall_metrics"][method]["tolerant_recall"] for m in comparison_metrics]
        ax.scatter(xs, ys, s=70, alpha=0.8, color=METHOD_COLORS[method], label=TAU_DISPLAY[method])
        for i,(x,y) in enumerate(zip(xs,ys),1):
            ax.text(x, y, f"Q{i}", fontsize=8, alpha=0.75)
    ax.set_xlabel("Tail/Head Ratio", fontweight="bold")
    ax.set_ylabel("Tolerant Recall@k", fontweight="bold")
    ax.set_title("Pareto: ranking-shape quality vs tolerant recall", fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Pareto plot → {out}")


def plot_headk_sweep(rows, out):
    if not rows: return
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[int(row["head_k"])][row["tau_method"]].append(float(row["tail_to_head_ratio"]))
    tau_labels = [TAU_DISPLAY[k] for k in TAU_METHOD_KEYS]
    head_ks = sorted(grouped.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    specs = [("tail_to_head_ratio","Mean Tail/Head Ratio","higher better"),
             ("tail_cv","Mean Tail CV","lower better"),
             ("tail_decay_rate","Mean Tail Decay","lower better")]
    all_grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        for col in ["tail_to_head_ratio","tail_cv","tail_decay_rate"]:
            all_grouped[int(row["head_k"])][row["tau_method"]][col].append(float(row[col]))
    for ax, (col, title, sub) in zip(axes, specs):
        for lbl in tau_labels:
            means = [np.mean(all_grouped[h][lbl][col]) if all_grouped[h][lbl][col] else np.nan for h in head_ks]
            stds  = [np.std(all_grouped[h][lbl][col])  if all_grouped[h][lbl][col] else 0.0  for h in head_ks]
            means = np.array(means); stds = np.array(stds)
            c = {TAU_DISPLAY[k]: METHOD_COLORS[k] for k in TAU_METHOD_KEYS}[lbl]
            ax.plot(head_ks, means, marker="o", lw=2, color=c, label=lbl)
            ax.fill_between(head_ks, means-stds, means+stds, color=c, alpha=0.15)
        ax.set_title(f"{title}\n({sub})", fontweight="bold"); ax.set_xlabel("HEAD_K"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Metric value"); axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"HEAD_K sweep plot → {out}")

# Main

def main(corpus_path, corpus_emb_cache=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    
    ids, titles, docs = load_corpus(corpus_path)
    
    corpus_cache = corpus_emb_cache or str(OUTPUT_DIR / "prompt_corpus_emb_nomic768.npy")
    C = build_embeddings(docs, cache_file=corpus_cache)

    queries = test_queries
    query_cache = str(OUTPUT_DIR / "prompt_query_emb_nomic768.npy")
    Q = build_embeddings(queries, cache_file=query_cache)


    tau_labels = [TAU_DISPLAY[k] for k in TAU_METHOD_KEYS]
    all_results = []
    comparison_metrics = []

    for qi, q in enumerate(queries):
        print(f"\n{'='*70}")
        print(f"Query {qi+1}: {q}")
        print("="*70)

        results = {}
        for tau_key in TAU_METHOD_KEYS:
            results[tau_key] = tau_search(C, Q[qi], TAU_VALUES[tau_key], k=RESULTS_K)

        min_len = min(len(results[k]) for k in TAU_METHOD_KEYS)
        for k_ in TAU_METHOD_KEYS:
            results[k_] = results[k_][:min_len]

        rc, rh, rt = results["Cosine"], results["Hybrid"], results["Taumode"]
        all_results.append((rc, rh, rt))

        # ranking agreement
        sc_h, kc_h = compute_ranking_metrics(rc, rh)
        sc_t, kc_t = compute_ranking_metrics(rc, rt)
        sh_t, kh_t = compute_ranking_metrics(rh, rt)

        # NDCG
        kn = min(NDCG_K, min_len)
        ndcg_hc = compute_ndcg(rh, rc, kn)
        ndcg_tc = compute_ndcg(rt, rc, kn)
        ndcg_th = compute_ndcg(rt, rh, kn)

        # tail
        tail_m = analyze_tail_distribution([rc, rh, rt], tau_labels,
                                            k_head=HEAD_K, k_tail=RESULTS_K)

        # recall
        gt_ids = [i for i,_ in rc]; gt_sc = [s for _,s in rc]
        recall_per_tau = {}
        for tau_key, res in zip(TAU_METHOD_KEYS, [rc, rh, rt]):
            recall_per_tau[tau_key] = compute_all_recall_metrics(
                [i for i,_ in res], [s for _,s in res], gt_ids, gt_sc)

        comparison_metrics.append({
            "query": q, "min_length": min_len,
            "spearman": (sc_h, sc_t, sh_t),
            "kendall":  (kc_h, kc_t, kh_t),
            "ndcg":     (ndcg_hc, ndcg_tc, ndcg_th),
            "tail_metrics": tail_m,
            "recall_metrics": recall_per_tau,
        })

        # print summary per query
        for lbl, res in zip(tau_labels, [rc, rh, rt]):
            print(f"\n{lbl}")
            for rank, (idx, score) in enumerate(res[:RESULTS_K], 1):
                print(f"  {rank:2d}. {titles[idx][:60]:<60} [{score:.4f}]")
        print(f"\n  Spearman C↔H={sc_h:.3f}  C↔T={sc_t:.3f}  H↔T={sh_t:.3f}")
        print(f"  NDCG@{kn}: H/C={ndcg_hc:.4f}  T/C={ndcg_tc:.4f}")
        print(f"  Recall summary:")
        for tk in TAU_METHOD_KEYS:
            rm = recall_per_tau[tk]
            sem = rm["semantic_recall"]
            print(f"    {tk:8s} trad={rm['traditional_recall']:.4f}  "
                  f"sem={'n/a' if (isinstance(sem,float) and math.isnan(sem)) else f'{sem:.4f}'}  "
                  f"tol={rm['tolerant_recall']:.4f}")

    
    save_search_results(queries, all_results, ids, titles,
                        OUTPUT_DIR / "prompt_search_results.csv")
    save_comparison_metrics(comparison_metrics,
                            OUTPUT_DIR / "prompt_comparison_metrics.csv")
    save_tail_metrics(comparison_metrics,
                      OUTPUT_DIR / "prompt_tail_metrics.csv")
    save_semantic_recall(comparison_metrics,
                         OUTPUT_DIR / "prompt_semantic_recall_metrics.csv")
    save_summary(comparison_metrics,
                 OUTPUT_DIR / "prompt_summary.csv")
    save_headk_sweep(run_headk_sweep(queries, all_results, HEAD_K_SWEEP),
                     OUTPUT_DIR / "prompt_headk_sweep.csv")
    save_run_metadata(OUTPUT_DIR / "prompt_run_metadata.json",
                      corpus_path, len(docs), len(queries), comparison_metrics)

    
    plot_top25(queries, all_results, ids, titles,
               OUTPUT_DIR / "prompt_top25_comparison.png")
    if all(min(len(r[0]),len(r[1]),len(r[2])) > HEAD_K for r in all_results):
        plot_tail_analysis(queries, all_results,
                           OUTPUT_DIR / "prompt_tail_analysis.png")
    plot_semantic_recall(comparison_metrics,
                         OUTPUT_DIR / "prompt_semantic_recall_comparison.png")
    plot_metric_deltas(comparison_metrics,
                       OUTPUT_DIR / "prompt_metric_deltas.png")
    valid_tail = [m for m in comparison_metrics if m["tail_metrics"]]
    if valid_tail:
        plot_win_loss_heatmap(valid_tail,
                              OUTPUT_DIR / "prompt_win_loss_heatmap.png")
        plot_pareto(valid_tail,
                    OUTPUT_DIR / "prompt_pareto_tradeoff.png")
    plot_headk_sweep(run_headk_sweep(queries, all_results, HEAD_K_SWEEP),
                     OUTPUT_DIR / "prompt_headk_sweep.png")

    
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"Average NDCG@{NDCG_K}:")
    print(f"  Hybrid vs Cosine : {np.mean([m['ndcg'][0] for m in comparison_metrics]):.4f}")
    print(f"  Taumode vs Cosine: {np.mean([m['ndcg'][1] for m in comparison_metrics]):.4f}")
    print("\nAverage Recall:")
    print(f"  {'Method':10s} {'Trad':>10} {'Semantic':>10} {'Tolerant':>10}")
    for tk in TAU_METHOD_KEYS:
        tv,sv,lv = [],[],[]
        for m in comparison_metrics:
            rm = m["recall_metrics"].get(tk)
            if rm:
                tv.append(rm["traditional_recall"])
                s = rm["semantic_recall"]
                if not (isinstance(s,float) and math.isnan(s)): sv.append(s)
                lv.append(rm["tolerant_recall"])
        print(f"  {tk:10s} {np.mean(tv):>10.4f} "
              f"{'n/a':>10}" if not sv else
              f"  {tk:10s} {np.mean(tv):>10.4f} {np.mean(sv):>10.4f} {np.mean(lv):>10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prompt corpus benchmark — nomic-embed-text-v1.5 @ 768d, "
                    "full CVE-benchmark metric suite.")
    parser.add_argument("--corpus",     required=True,
                        help="Path to prompt corpus (.jsonl/.json/.txt or directory)")
    parser.add_argument("--corpus-emb", default=None,
                        help="Path to pre-computed corpus .npy (optional, speeds up re-runs)")
    args = parser.parse_args()
    main(args.corpus, args.corpus_emb)