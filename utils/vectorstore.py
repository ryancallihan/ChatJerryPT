from typing import List, Callable, Tuple

import numpy as np
from hyperdb import HyperDB


"""
Initialises vector store and sets up serving funcs

Currently, will just be a wrapper for HyperDB, but am abstracting 
for use with other stores in future.
"""

class Vectorstore:
    
    def __init__(
            self,
            data: List[dict] = None,
            vectors: List[np.ndarray] = None,
            embedding_func: Callable = None
        ) -> None:
        
        self.db = HyperDB()
        
        if data is not None and vectors is not None and embedding_func is not None:
            self.db = HyperDB(
                    documents=data, 
                    vectors=vectors, 
                    embedding_function=embedding_func
                )
            
            self.db.save("data/vectorstore.pkl")
        
    def load(self, path: str = "data/vectorstore.pkl") -> None:
        self.db = self.db.load(path)
        
    def query(self, document: str, top_k: int) -> List[Tuple[dict, float]]:
        return self.db.query(document, top_k=top_k)


