import os
import faiss
from typing import Tuple
import numpy as np

def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index(index: faiss.IndexFlatL2,index_path: str):
    os.makedirs(os.path.dirname(index_path),exist_ok=True)
    faiss.write_index(index,index_path)

def load_index(index_path: str) -> faiss.IndexFlatL2:
    return faiss.read_index(index_path)

def save_metadata(metadata,path:str):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    np.save(path,metadata,allow_pickle=True)

def load_metadata(path:str):
    return np.load(path,allow_pickle=True).tolist()