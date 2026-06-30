from typing import List
from groq import Groq

class QueryWritter:
    def __init__(self,apikey:str,model:str):
        self.client = Groq(api_key=apikey)
        self.model=model
    
    def _call_llm(self,prompt:str,temperature:float=0.3) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    
    def rewrite(self,question:str,n:int=3) -> List[str]:
        prompt=f"""
You are a query rewrite assistant

rewrite the following question into {n} differenct but equivalent search queries

Original Question:
{question}

return each rewritten query on a new line, no numbering, no extra text
"""
        out = self._call_llm(prompt=prompt,temperature=0.3)
        lines = [l.strip() for l in out.split("\n") if l.strip()]
        seen = []
        for l in lines:
            if l not in seen:
                seen.append(l)
        return seen