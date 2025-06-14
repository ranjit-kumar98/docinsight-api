import os
import pickle
import numpy as np
import faiss
from typing import List, Tuple, Optional

VECTOR_SIZE = 384  
INDEX_PATH = "faiss.index"
META_PATH  = "faiss_metadata.pkl"

class FaissStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(VECTOR_SIZE)
        self.metadata = []  # list of (doc_id, chunk_text)
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                self.metadata = pickle.load(f)

    def add(self, vectors: List[List[float]], metadatas: List[Tuple[str, str]]):
        arr = np.array(vectors).astype("float32")
        self.index.add(arr)
        self.metadata.extend(metadatas)
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

    def search(self, query_vector: List[float], top_k: int = 5, doc_id: Optional[str] = None) -> List[Tuple[float, str]]:
        query_np = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(query_np, top_k * 5)  # overfetch for filtering
        results = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            meta_doc_id, chunk = self.metadata[idx]
            if doc_id is None or meta_doc_id == doc_id:
                score = 1 / (1 + dist)  # convert L2 distance to pseudo similarity
                results.append((score, chunk))
            if len(results) >= top_k:
                break

        return results

