import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(result: list):
    embeddings_matrix = np.array(result)
    similarity_matrix = cosine_similarity(embeddings_matrix)
    return similarity_matrix
