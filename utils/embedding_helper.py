# utils/embedding_helper.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load once
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return model.encode(text)

def calculate_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

def calculate_similarity_batch(embedding, embedding_list):
    similarities = cosine_similarity([embedding], embedding_list)
    return similarities[0]
