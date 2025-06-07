from numpy import dot
from numpy.linalg import norm

def cosine_similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        print("Error: Vectors must be of the same length for cosine similarity.")
        raise ValueError("Vectors must be of the same length for cosine similarity.")
        
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))
