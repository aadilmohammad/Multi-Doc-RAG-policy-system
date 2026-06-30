from typing import List,Dict,Any

import numpy as np
from groq import Groq
from rag.config import RAGConfig
from rag.query_rewrite import QueryWritter
from rag.retriever import Retriever

class RAGPipeline:
    def __init__(self,config:RAGConfig,retriever:Retriever):
        self.config = config
        self.rewriter = QueryWritter(apikey=config.groq_api_key,model=config.groq_model)
        self.client = Groq(api_key=config.groq_api_key)
        self.retriever=retriever

    def _call_llm(self,prompt:str,temperature:float=0.0) -> str:
        res = self.client.chat.completions.create(
            model=self.config.groq_model,
            messages=[{"role":"user","content":prompt}],
            temperature=temperature
        )
        return res.choices[0].message.content.strip()
    
    def _build_context(self,chunks:List[Dict[str,Any]],max_chunks:int) -> str:
        selected = chunks[:max_chunks]
        blocks = []
        for r in selected:
            src = r["metadata"]["source"]
            blocks.append(f"{src}\n{r["chunk"]}")
        return "\n\n------\n\n".join(blocks)
    
    def answer(self,question:str) -> Dict[str,Any]:
        rewrites = self.rewriter.rewrite(question,n=3)
        retrieved = self.retriever.retrieve_multi_query(
            question,
            rewrites,
            k_per_query=self.config.k_per_query
        )

        context = self._build_context(retrieved,max_chunks=self.config.max_chunks_for_context)

        prompt = f"""
You are a helpful assistant

Use only the context below to provide the answer to the question

Question:
{question}

Context:
{context}

Rules:
- If the answer is not in the context, say you cannot find it in the documents.
- Be concise and clear.
"""
        answer = self._call_llm(prompt=prompt,temperature=0.0)
        return {
            "question":question,
            "answer":answer,
            "retrieved":retrieved,
            "context":context
        }