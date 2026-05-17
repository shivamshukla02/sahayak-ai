import os
import chromadb
from chromadb.utils import embedding_functions

# Configure file paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BACKEND_DIR, "documents")
CHROMA_DIR = os.path.join(BACKEND_DIR, "chroma_db")

def parse_and_chunk_documents():
    """Reads all txt files from the documents folder and splits them into paragraphs."""
    chunks = []
    metadata_list = []
    ids = []
    
    if not os.path.exists(DOCS_DIR):
        print(f"[ERROR] Documents directory does not exist: {DOCS_DIR}")
        return chunks, metadata_list, ids
        
    global_id_counter = 1
    
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_DIR, filename)
            print(f"[INFO] Reading document: {filename}")
            
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Split content by double newlines to isolate separate paragraphs
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            
            for idx, para in enumerate(paragraphs):
                chunks.append(para)
                metadata_list.append({
                    "source": filename,
                    "paragraph_index": idx
                })
                ids.append(f"doc_{filename.replace('.txt', '')}_{global_id_counter}")
                global_id_counter += 1
                
    return chunks, metadata_list, ids

def main():
    print("--------------------------------------------------")
    print("  SAHAYAK AI -- LOCAL DOCUMENT VECTOR BUILDER     ")
    print("--------------------------------------------------")
    
    # 1. Parse documents
    chunks, metadata, ids = parse_and_chunk_documents()
    if not chunks:
        print("[WARNING] No documents found or chunked. Please add files to backend/documents/")
        return
        
    print(f"[SUCCESS] Parsed {len(chunks)} text chunks to index.")
    
    # 2. Setup Persistent ChromaDB Client
    print(f"[INFO] Connecting to Persistent ChromaDB at: {CHROMA_DIR}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # 3. Use standard sentence-transformer embedding model (runs completely offline after first model download)
    print("[INFO] Instantiating SentenceTransformer Embedding Function (all-MiniLM-L6-v2)...")
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # 4. Create or recreate collection
    collection_name = "emergency_manuals"
    try:
        # Delete existing collection to avoid duplication
        chroma_client.delete_collection(name=collection_name)
        print(f"[INFO] Cleared pre-existing collection '{collection_name}' for rebuilding.")
    except Exception:
        pass
        
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}  # Cosine distance metric for RAG similarity
    )
    
    # 5. Insert documents in batches
    print(f"[INFO] Indexing documents into ChromaDB...")
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        collection.add(
            documents=chunks[i:end_idx],
            metadatas=metadata[i:end_idx],
            ids=ids[i:end_idx]
        )
        print(f"  Indexed batch {i//batch_size + 1}: Chunks {i} to {end_idx - 1}")
        
    print("--------------------------------------------------")
    print(f"[SUCCESS] ChromaDB Build Complete! Total Chunks Indexed: {collection.count()}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
