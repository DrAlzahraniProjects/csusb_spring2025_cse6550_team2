import json
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS

def main():
    # 1. Load the JSON data
    try:
        with open("../data/output.json", "r") as f: data: list[dict[str, str | list[str]]] = json.load(f)
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return

    # Extract all text segments from the JSON file.
    # segments: list[str] = []
    # urls: list[dict[str, str]] = []
    # for page in data:
    #     currentSegments: list[str] = page.get("segments", [])
    #     segments += currentSegments
    #     currentUrl: str = page.get("url", "")
    #     urls += [{"url": currentUrl} for _ in currentSegments]

    segments = [segment for page in data for segment in page.get("segments", [])]
    urls = [{"url": page.get("url", "")} for page in data for _ in page.get("segments", [])]
    print(f"Total segments loaded: {len(segments)}")

    if not segments:
        print("No segments found. Exiting.")
        return

    # 2. Generate embeddings using a pre-trained SentenceTransformer model.
    embeddings = FastEmbedEmbeddings()
    print("Embeddings generated.")

    # 3. Build a FAISS index.
    vectorstore = FAISS.from_texts(segments, embeddings, urls)
    print(f"FAISS index built.")

    # 4. Save the FAISS index to disk.
    vectorstore.save_local("../data/index")
    print(f"FAISS index saved at ../data/index.")

if __name__ == "__main__":
    main()
