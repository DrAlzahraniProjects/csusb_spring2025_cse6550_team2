import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def main():
    # 1. Load the JSON data
    try:
        with open("../data/output.json", "r") as f: data: list[dict[str, str]] = json.load(f)
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return

    # Extract all text segments from the JSON file.
    # segments: list[str] = ["".join(s for s in item.get("segments", [])) for item in data]
    segments: list[str] = [segment for page in data for segment in page.get("segments", [])]
    print(f"Total segments loaded: {len(segments)}")

    if not segments:
        print("No segments found. Exiting.")
        return

    # 2. Generate embeddings using a pre-trained SentenceTransformer model.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embeddings generated.")

    # 3. Build a FAISS index.
    vectorstore = FAISS.from_texts(segments, embeddings)
    print(f"FAISS index built.")

    # 4. Save the FAISS index to disk.
    vectorstore.save_local("../data/index")
    print(f"FAISS index saved at ../data/index.")

if __name__ == "__main__":
    main()
