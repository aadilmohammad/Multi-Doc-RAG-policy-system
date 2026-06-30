import os
import glob
from typing import Dict

def load_documents(folder: str) -> Dict[str, str]:
    docs={}
    os.makedirs(folder,exist_ok=True)
    for pth in glob.glob(os.path.join(folder,"*")):
        if os.path.isfile(pth) and  pth.lower().endswith((".txt",".md")):
            with open(pth, "r", encoding="utf-8", errors="ignore") as f:
                docs[os.path.basename(pth)] = f.read()
    return docs