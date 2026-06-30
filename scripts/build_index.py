import os
import numpy as np



from rag.ingestion import load_documents
from rag.index import save_index,build_index,save_metadata
from rag.config import RAGConfig
from rag.chunking import chunk_text
from rag.models import ChunkMetadata
from rag.embeddings import EmbeddingModel

def main():
    cfg = RAGConfig()

    docs = load_documents(cfg.docs_folder)
    if not docs:
        print("No Documents Found. Please add Docs and rerun")
        return
    
    chunks = []
    metadata = []
    chunk_id = 0

    for doc_name,text in docs.items():
        doc_chunks = chunk_text(text=text,max_tokens=cfg.chunk_size,overlap=cfg.chunk_overlap)
        for i, ch in enumerate(doc_chunks):
            chunks.append(ch)
            meta = ChunkMetadata(chunk_id,source=doc_name,chunk_index=i).to_dict()
            metadata.append(meta)
            chunk_id+=1

    embed_model = EmbeddingModel(model_name=cfg.embed_Model_name)
    embeddings = embed_model.encode(chunks)

    index = build_index(embeddings)

    os.makedirs("./data",exist_ok=True)
    save_index(index,cfg.index_path)
    save_metadata({"chunks":chunks,"metadata":metadata},cfg.metadata_path)

    print(f"Built index with {len(chunks)} chunks.")


if __name__ == "__main__":
    main()

        

    