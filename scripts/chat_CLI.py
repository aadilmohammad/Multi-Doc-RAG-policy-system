import numpy as np

from rag.config import RAGConfig
from rag.embeddings import EmbeddingModel
from rag.index import load_index,load_metadata
from rag.retriever import Retriever
from rag.rag_pipeline import RAGPipeline
from rag.evaluation import Evaluator

def main():
    cfg = RAGConfig()

    index = load_index(cfg.index_path)
    metadat_obj = load_metadata(cfg.metadata_path)
    chunks = metadat_obj["chunks"]
    metadata = metadat_obj["metadata"]

    embed_model = EmbeddingModel(model_name=cfg.embed_Model_name)
    retriever = Retriever(index=index,embed_model=embed_model,chunks=chunks,metadata=metadata)
    pipeline = RAGPipeline(config=cfg,retriever=retriever)
    evaluator = Evaluator(cfg=cfg)

    print("Full Rag system (Query Rewrite + Multi query doc) type quit or exit to end chat:\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit","quit"):
            print("\nAssistant: Bye.")
            return
    
        res = pipeline.answer(q)
        print("\nAssistant: ",res["answer"])
        print("\nSources: ")
        seen = set()
        for r in res["retrieved"][:5]:
            src = r["metadata"]["source"]
            if  src not in seen:
                print("\n-",src)
                seen.add(src)

        eval_json = evaluator.evaluate_answer(question=res["question"],context=res["context"],answer=res["answer"])
        print("\n[Eval]:",eval_json)
        print("\n"+"=*60"+"\n")
    
if __name__ == "__main__":
    main()