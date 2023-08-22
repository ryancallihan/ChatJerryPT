from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


"""
Loads (and downloads) huggingface sentence transformer and sets up embedding func 
"""


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME)

def embedding_function(documents: List[str]) -> List[np.ndarray]:
    """Embedding function. Abstracting it so that I can add 
    more embedding options in future.
    
    TODO: This is not optimal for larger sets of docs

    Args:
        documents (List[str]): list of documents to embed

    Returns:
        List[np.ndarray]: list of embeddings
    """
    return EMBEDDING_MODEL.encode(documents)
