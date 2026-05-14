
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import json


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
data_path = DATA_DIR / "dataset.json"
output_path = DATA_DIR / "embeddings.npy"

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def extract_nl(element):
    return f"{element['title']}\n{element['content']}" # We start with the most naive approach: just concatenate the title and content. We can experiment with more complex approaches later.

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