
import tensorflow as tf
from scipy.spatial.distance import cosine
from signver.matcher.faiss_index import FaissIndex


class Matcher():
    def __init__(self, index_dim=256):
        self.index = FaissIndex(index_dim)

    def cosine_distance(self, vector_1, vector_2):
        return cosine(vector_1.flatten(), vector_2.flatten())

    def verify(self, vector_1, vector_2, threshold=0.18) -> bool:
        distance = self.cosine_distance(
            vector_1.flatten(), vector_2.flatten())
        return distance < threshold

    def identify(query, index):
        pass

    def load_index(self, model_path: str):
        self.index.load()

    def save_index(self, save_path: str):
        pass
