import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

@dataclass
class RAGConfig:
    docs_folder: str = "./docs"
    index_path: str = ".data/faiss.index"
    metadata_path: str = ".data/metadata.npy"
    embed_Model_name: str = "all-MiniLM-L6-v2"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str = os.getenv("GROQ_API_KEY","")
    chunk_size: int = 160
    chunk_overlap: int = 40
    k_per_query: int = 5
    max_chunks_for_context: int = 6
