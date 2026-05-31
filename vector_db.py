import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

INDEX_PATH = "vector_db.pkl"

class LSHVectorIndex:
    def __init__(self, dimension, num_hyperplanes=8, seed=42):
        self.dimension = dimension
        self.num_hyperplanes = num_hyperplanes
        np.random.seed(seed)
        # Generate random projection vectors (hyperplanes)
        self.hyperplanes = np.random.randn(num_hyperplanes, dimension)
        self.buckets = {}  # hash_string -> list of item IDs
        self.vector_store = {}  # item_id -> original vector

    def hash_vector(self, vector):
        v = np.array(vector)
        # Project vector onto hyperplanes
        projections = np.dot(self.hyperplanes, v)
        # Convert signs to a binary hash string
        binary_hash = "".join(["1" if x >= 0 else "0" for x in projections])
        return binary_hash

    def insert(self, item_id, vector):
        if vector is None:
            return
        # Ensure correct key type
        item_id = str(item_id)
        # If item already exists, clean up old references
        self.remove(item_id)
        
        binary_hash = self.hash_vector(vector)
        if binary_hash not in self.buckets:
            self.buckets[binary_hash] = []
        self.buckets[binary_hash].append(item_id)
        self.vector_store[item_id] = np.array(vector)

    def remove(self, item_id):
        item_id = str(item_id)
        if item_id in self.vector_store:
            del self.vector_store[item_id]
        for bucket in self.buckets.values():
            if item_id in bucket:
                bucket.remove(item_id)

    def query(self, query_vector, max_hamming_distance=2):
        if query_vector is None or not self.vector_store:
            return []
        
        query_hash = self.hash_vector(query_vector)
        candidates = set()
        
        # Retrieve candidate IDs from close hash buckets
        for bucket_hash, item_ids in self.buckets.items():
            hamming_dist = sum(c1 != c2 for c1, c2 in zip(query_hash, bucket_hash))
            if hamming_dist <= max_hamming_distance:
                candidates.update(item_ids)
                
        # If no candidates are found, fall back to evaluating all stored items
        if not candidates:
            candidates = set(self.vector_store.keys())
            
        results = []
        query_v = np.array(query_vector)
        
        for item_id in candidates:
            if item_id not in self.vector_store:
                continue
            cand_v = self.vector_store[item_id]
            norm_q = np.linalg.norm(query_v)
            norm_c = np.linalg.norm(cand_v)
            
            if norm_q > 0 and norm_c > 0:
                sim = np.dot(query_v, cand_v) / (norm_q * norm_c)
            else:
                sim = 0.0
                
            results.append((item_id, float(sim)))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results

class LocalVectorDB:
    def __init__(self, image_dim=1280, text_dim=128):
        self.image_dim = image_dim
        self.text_dim = text_dim
        
        # Use 12 hyperplanes for high-dimensional images (4096 buckets)
        self.image_index = LSHVectorIndex(image_dim, num_hyperplanes=12, seed=42)
        # Use 8 hyperplanes for text (256 buckets)
        self.text_index = LSHVectorIndex(text_dim, num_hyperplanes=8, seed=101)
        
        # Stateless Hashing Vectorizer to produce consistent 128-dimensional dense vectors
        self.vectorizer = HashingVectorizer(n_features=text_dim, alternate_sign=False)

    def get_text_vector(self, text):
        # Convert text to a dense 128-dimensional array
        return self.vectorizer.transform([text]).toarray()[0]

    def insert_item(self, item_id, image_vector, text_string):
        item_id = str(item_id)
        if image_vector is not None:
            self.image_index.insert(item_id, image_vector)
        if text_string:
            text_vector = self.get_text_vector(text_string)
            self.text_index.insert(item_id, text_vector)
        self.save()

    def remove_item(self, item_id):
        item_id = str(item_id)
        self.image_index.remove(item_id)
        self.text_index.remove(item_id)
        self.save()

    def save(self):
        try:
            with open(INDEX_PATH, "wb") as f:
                pickle.dump({
                    "image_buckets": self.image_index.buckets,
                    "image_store": self.image_index.vector_store,
                    "text_buckets": self.text_index.buckets,
                    "text_store": self.text_index.vector_store
                }, f)
            print("test msg : Saved local vector DB successfully.")
        except Exception as e:
            print(f"Error saving local vector DB: {e}")

    def load(self):
        if not os.path.exists(INDEX_PATH):
            return False
        try:
            with open(INDEX_PATH, "rb") as f:
                data = pickle.load(f)
                self.image_index.buckets = data.get("image_buckets", {})
                self.image_index.vector_store = data.get("image_store", {})
                self.text_index.buckets = data.get("text_buckets", {})
                self.text_index.vector_store = data.get("text_store", {})
            print("test msg : Loaded local vector DB successfully.")
            return True
        except Exception as e:
            print(f"Error loading local vector DB: {e}")
            return False

# Global instance of the local vector database
vector_db = LocalVectorDB()
vector_db.load()
