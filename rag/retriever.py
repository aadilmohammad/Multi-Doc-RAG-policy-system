from typing import List,Dict,Any
import numpy as np
import faiss

from rag.embeddings import EmbeddingModel

class Retriever:
    def __init__(self,index:faiss.IndexFlatL2, embed_model:EmbeddingModel,chunks,metadata):
        self.index = index
        self.embed_model = embed_model
        self.chunks = chunks
        self.metadata = metadata

    def retrieve_single(self,query:str,k:int=5) -> List[Dict[str,Any]]:
        q_emb = self.embed_model.encode([query]).reshape(1,-1)
        D, I = self.index.search(q_emb,k)
        results = []
        for score,idx in zip(D[0],I[0]):
            results.append({
                "score": score,
                "chunk": self.chunks[idx],
                "metadata": self.metadata[idx]
            })
        return results
    
    def retrieve_multi_query(self,question:str,rewrites:List[str],k_per_query:int=5) -> List[Dict[str,Any]]:
        all_results = []
        for q in [question] + rewrites:
            res = self.retrieve_single(q)
            all_results.extend(res)

        best_by_id: Dict[int,Dict[str,Any]] = {}
        for r in all_results:
            cid = r["metadata"]["id"]
            if cid not in best_by_id or best_by_id[cid]["score"] > r["score"]:
                best_by_id[cid] = r
            
        merged_list = list(best_by_id.values())
        merged_list.sort(key=lambda x:x["score"])
        return merged_list
