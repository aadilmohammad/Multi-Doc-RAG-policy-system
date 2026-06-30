from typing import List

def chunk_text(text:str,max_tokens:int,overlap:int) -> List[str]:
    words=text.split()
    chunks=[]
    start = 0

    while start < len(words):
        end = start + max_tokens
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    
    return chunks
