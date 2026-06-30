from dataclasses import dataclass
from typing import Dict,Any

@dataclass
class ChunkMetadata:
    id:int
    source:str
    chunk_index:int

    def to_dict(self) -> Dict[str,Any]:
        return {
            "id": self.id,
            "source": self.source,
            "chunk_index":self.chunk_index
        }