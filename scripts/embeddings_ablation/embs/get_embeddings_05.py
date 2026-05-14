
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm



ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
data_path = DATA_DIR / "dataset.json"
output_path = DATA_DIR / "embeddings_06.npy"

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_nl(element):  #more structured extraction
    placeholders_str = (
        f"Template variables: {', '.join(element['placeholders'])}."
        if element['has_placeholders'] and element['placeholders']
        else "No template variables."
    )

    corpus = f"""Title: {element['title']}

Category: {element['category']} > {element['subcategory']}
Difficulty: {element['difficulty']}
Target model: {element.get('target_model', 'general')}
Tags: {', '.join(element['tags'])}
{placeholders_str}

Content:
{element['content']}
"""
    return corpus


def main():
    
    print("Loading dataset...")
    corpus = load_json(data_path)
    
    
    print(f"Extracting text from {len(corpus)} items...")
    sentences = [extract_nl(item) for item in corpus]

    
    print("Loading model...")
    model = SentenceTransformer('perplexity-ai/pplx-embed-v1-0.6b', trust_remote_code=True)

    
    
    print("Starting multi-process pool")
    pool = model.start_multi_process_pool()

    
    print("Generating embeddings")
    
    
    chunk_size = 1000 
    all_embeddings = []
    
    
    for i in tqdm(range(0, len(sentences), chunk_size), desc="Generazione Embeddings"):
        chunk = sentences[i : i + chunk_size]
        
    
        chunk_embeddings = model.encode_multi_process(chunk, pool)
        all_embeddings.extend(chunk_embeddings)

    
    embeddings = np.array(all_embeddings)

    
    model.stop_multi_process_pool(pool)

    
    print(f"Saving embeddings of shape {embeddings.shape} to disk...")
    embs_np = embeddings.astype(np.float64)
    np.save(output_path, embs_np)
    
    print(f"Done! Saved to {output_path}")

if __name__ == '__main__':
    main()