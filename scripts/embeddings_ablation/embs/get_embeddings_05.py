
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
from semantic_engine_demo.json_load import load_json
from tqdm import tqdm


# Setup paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
data_path = DATA_DIR / "dataset.json"
output_path = DATA_DIR / "embeddings_06.npy"


def extract_nl(element):
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
    # 1. Load the data
    print("Loading dataset...")
    corpus = load_json(data_path)
    
    # 2. Extract all text into a single list
    print(f"Extracting text from {len(corpus)} items...")
    sentences = [extract_nl(item) for item in corpus]

    # 3. Initialize the model ONCE
    print("Loading model...")
    model = SentenceTransformer('perplexity-ai/pplx-embed-v1-0.6b', device='cpu', trust_remote_code=True)

    # 4. Start the multi-processing pool
    # This automatically detects how many CPU cores you have
    print("Starting multi-process pool...")
    pool = model.start_multi_process_pool()

    # 5. Encode the sentences in parallel A BLOCCHI per monitorare i progressi
    print("Generating embeddings... (this might take a moment)")
    
    # Decidi quanti elementi processare per volta (es. 1000, o 5000)
    chunk_size = 1000 
    all_embeddings = []
    
    # Ciclo con tqdm per la barra di progresso
    for i in tqdm(range(0, len(sentences), chunk_size), desc="Generazione Embeddings"):
        chunk = sentences[i : i + chunk_size]
        
        # Codifica solo questo blocco
        chunk_embeddings = model.encode_multi_process(chunk, pool)
        all_embeddings.extend(chunk_embeddings)

    # Trasformiamo la lista finale in un array numpy
    embeddings = np.array(all_embeddings)

    # 6. Stop the pool to free up CPU resources
    model.stop_multi_process_pool(pool)

    # 7. Convert to float64 and save
    print(f"Saving embeddings of shape {embeddings.shape} to disk...")
    embs_np = embeddings.astype(np.float64)
    np.save(output_path, embs_np)
    
    print(f"Done! Saved to {output_path}")

if __name__ == '__main__':
    main()