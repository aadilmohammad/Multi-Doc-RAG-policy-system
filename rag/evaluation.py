from groq import Groq
from rag.config import RAGConfig

class Evaluator:
    def __init__(self,cfg:RAGConfig):
        self.client = Groq(api_key=cfg.groq_api_key)
        self.model = cfg.groq_model

    def evaluate_answer(self,question,context,answer) -> str:
        prompt = f"""
You are an evaluation judge

Evaluate the following answer based on given context

Question:
{question}

Answer:
{answer}

Context:
{context}

Rate the answer on:
- Faithfulness (0-1)
- Groundedness (0-1)
- Relevance (0-1)
- Completeness (0-1)

Return a JSON object with keys:
faithfulness, groundedness, relevance, completeness.
"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0
        )
        return resp.choices[0].message.content.strip()