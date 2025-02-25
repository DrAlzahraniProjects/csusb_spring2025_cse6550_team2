import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import SCRAPED_DATA_PATH, FAISS_INDEX_PATH

def main():
    # 1. Load the JSON data
    try:
        with open(SCRAPED_DATA_PATH, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return

    # Extract all text segments from the JSON file.
    segments = []
    for item in data:
        segments.extend(item.get("segments", []))
    print(f"Total segments loaded: {len(segments)}")


    filtered_segments = [
        s for s in segments
        if len(s)> 50 and "study abroad" in s.lower() and "csusb" in s.lower()
    ]

    if not segments:
        print("No segments found. Exiting.")
        return

    # 2. Generate embeddings using a pre-trained SentenceTransformer model.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(segments)
    print("Embeddings generated.")

    # 3. Normalize embeddings to unit length (for cosine similarity).
    embeddings = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    # 4. Build a FAISS index using inner product search.
    # With normalized embeddings, inner product is equivalent to cosine similarity.
    embedding_dim = normalized_embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)  # Using inner product.
    index.add(normalized_embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")

    # 5. Save the FAISS index to disk.
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved at '{FAISS_INDEX_PATH}'.")

if __name__ == "__main__":
    main()
