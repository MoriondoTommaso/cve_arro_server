
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import json

# Setup paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
data_path = DATA_DIR / "dataset.json"
output_path = DATA_DIR / "embeddings_02.npy"

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def extract_nl(element):
    return element # we try to embed the whole element as a string, to see if the model can learn to ignore the irrelevant fields. This is a very naive approach, but it will serve as a baseline for our ablation study.

def main():
    print("Loading dataset...")
    corpus = load_json(data_path)
    
    print(f"Extracting text from {len(corpus)} items...")
    sentences = [extract_nl(item) for item in corpus]

    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Starting multi-process pool...")
    pool = model.start_multi_process_pool()

    
    print("Generating embeddings")
    embeddings = model.encode(sentences, pool)

    model.stop_multi_process_pool(pool)

    
    print(f"Saving embeddings of shape {embeddings.shape} to disk...")
    embs_np = np.array(embeddings).astype(np.float64)
    np.save(output_path, embs_np)
    
    print(f"Done! Saved to {output_path}")

if __name__ == '__main__':
    main()