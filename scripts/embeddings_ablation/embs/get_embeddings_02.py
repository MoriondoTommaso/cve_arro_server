
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
from semantic_engine_demo.json_load import load_json

# Setup paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
data_path = DATA_DIR / "dataset.json"
output_path = DATA_DIR / "embeddings_02.npy"


def extract_nl(element):
    return element

def main():
    # 1. Load the data
    print("Loading dataset...")
    corpus = load_json(data_path)
    
    # 2. Extract all text into a single list
    print(f"Extracting text from {len(corpus)} items...")
    sentences = [extract_nl(item) for item in corpus]

    # 3. Initialize the model ONCE
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    # 4. Start the multi-processing pool
    # This automatically detects how many CPU cores you have
    print("Starting multi-process pool...")
    pool = model.start_multi_process_pool()

    # 5. Encode the sentences in parallel
    print("Generating embeddings... (this might take a moment)")

    embeddings = model.encode_multi_process(sentences, pool)

    # 6. Stop the pool to free up CPU resources
    model.stop_multi_process_pool(pool)

    # 7. Convert to float64 (optional, but requested in your earlier traceback)
    # and save to a .npy file
    print(f"Saving embeddings of shape {embeddings.shape} to disk...")
    embs_np = np.array(embeddings).astype(np.float64)
    np.save(output_path, embs_np)
    
    print(f"Done! Saved to {output_path}")

# This is strictly required for Python multiprocessing to work safely
if __name__ == '__main__':
    main()