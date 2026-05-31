import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

INDEX_PATH = "vector_db.db"

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
        
        # Generate all binary string neighbors within max_hamming_distance Hamming distance
        n = len(query_hash)
        chars = list(query_hash)
        
        def generate(index, current_chars, current_dist):
            if current_dist > max_hamming_distance:
                return
            if index == n:
                h_str = "".join(current_chars)
                if h_str in self.buckets:
                    candidates.update(self.buckets[h_str])
                return
            
            # Scenario 1: Keep bit as is
            generate(index + 1, current_chars, current_dist)
            
            # Scenario 2: Flip bit
            orig = current_chars[index]
            current_chars[index] = "1" if orig == "0" else "0"
            generate(index + 1, current_chars, current_dist + 1)
            current_chars[index] = orig # backtrack
            
        generate(0, chars, 0)
                
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
    def __init__(self, image_dim=1280, text_dim=1024, db_path="vector_db.db"):
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.db_path = db_path
        
        # Use 12 hyperplanes for high-dimensional images (4096 buckets)
        self.image_index = LSHVectorIndex(image_dim, num_hyperplanes=12, seed=42)
        # Use 8 hyperplanes for text (256 buckets)
        self.text_index = LSHVectorIndex(text_dim, num_hyperplanes=8, seed=101)
        
        # Stateless Hashing Vectorizer to produce consistent 1024-dimensional dense vectors
        self.vectorizer = HashingVectorizer(n_features=text_dim, alternate_sign=False)
        self._init_db()

    def _init_db(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_buckets (
                hash_string TEXT,
                item_id TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_store (
                item_id TEXT PRIMARY KEY,
                vector BLOB
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS text_buckets (
                hash_string TEXT,
                item_id TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS text_store (
                item_id TEXT PRIMARY KEY,
                vector BLOB
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_img_buckets_hash ON image_buckets(hash_string)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_img_buckets_item ON image_buckets(item_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_txt_buckets_hash ON text_buckets(hash_string)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_txt_buckets_item ON text_buckets(item_id)")
        conn.commit()
        conn.close()

    def get_text_vector(self, text):
        # Convert text to a dense 1024-dimensional array
        return self.vectorizer.transform([text]).toarray()[0]

    def insert_item(self, item_id, image_vector, text_string):
        import sqlite3
        item_id = str(item_id)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM image_buckets WHERE item_id = ?", (item_id,))
            cursor.execute("DELETE FROM image_store WHERE item_id = ?", (item_id,))
            cursor.execute("DELETE FROM text_buckets WHERE item_id = ?", (item_id,))
            cursor.execute("DELETE FROM text_store WHERE item_id = ?", (item_id,))

            if image_vector is not None:
                image_vector = np.array(image_vector, dtype=np.float32)
                # Normalize Visual Vector before indexing/storing
                norm = np.linalg.norm(image_vector)
                if norm > 0:
                    image_vector = image_vector / norm
                
                self.image_index.insert(item_id, image_vector)
                image_hash = self.image_index.hash_vector(image_vector)
                cursor.execute("INSERT INTO image_buckets (hash_string, item_id) VALUES (?, ?)", (image_hash, item_id))
                cursor.execute("INSERT INTO image_store (item_id, vector) VALUES (?, ?)", (item_id, image_vector.tobytes()))

            if text_string:
                text_vector = self.get_text_vector(text_string)
                text_vector = np.array(text_vector, dtype=np.float32)
                
                self.text_index.insert(item_id, text_vector)
                text_hash = self.text_index.hash_vector(text_vector)
                cursor.execute("INSERT INTO text_buckets (hash_string, item_id) VALUES (?, ?)", (text_hash, item_id))
                cursor.execute("INSERT INTO text_store (item_id, vector) VALUES (?, ?)", (item_id, text_vector.tobytes()))

            conn.commit()
            print(f"test msg : Saved item {item_id} to SQLite local vector DB.")
        except Exception as e:
            conn.rollback()
            print(f"Error inserting item into SQLite: {e}")
        finally:
            conn.close()

    def remove_item(self, item_id):
        import sqlite3
        item_id = str(item_id)
        self.image_index.remove(item_id)
        self.text_index.remove(item_id)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM image_buckets WHERE item_id = ?", (item_id,))
            cursor.execute("DELETE FROM image_store WHERE item_id = ?", (item_id,))
            cursor.execute("DELETE FROM text_buckets WHERE item_id = ?", (item_id,))
            cursor.execute("DELETE FROM text_store WHERE item_id = ?", (item_id,))
            conn.commit()
            print(f"test msg : Removed item {item_id} from SQLite local vector DB.")
        except Exception as e:
            conn.rollback()
            print(f"Error removing item from SQLite: {e}")
        finally:
            conn.close()

    def save(self):
        # Kept for backwards compatibility, actual saves are atomic and transactional
        pass

    def load(self):
        import sqlite3
        if not os.path.exists(self.db_path):
            return False
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load image buckets
            cursor.execute("SELECT hash_string, item_id FROM image_buckets")
            image_buckets = {}
            for h_str, i_id in cursor.fetchall():
                if h_str not in image_buckets:
                    image_buckets[h_str] = []
                image_buckets[h_str].append(i_id)
            self.image_index.buckets = image_buckets
            
            # Load image vectors
            cursor.execute("SELECT item_id, vector FROM image_store")
            image_store = {}
            for i_id, vec_blob in cursor.fetchall():
                image_store[i_id] = np.frombuffer(vec_blob, dtype=np.float32)
            self.image_index.vector_store = image_store
            
            # Load text buckets
            cursor.execute("SELECT hash_string, item_id FROM text_buckets")
            text_buckets = {}
            for h_str, i_id in cursor.fetchall():
                if h_str not in text_buckets:
                    text_buckets[h_str] = []
                text_buckets[h_str].append(i_id)
            self.text_index.buckets = text_buckets
            
            # Load text vectors
            cursor.execute("SELECT item_id, vector FROM text_store")
            text_store = {}
            for i_id, vec_blob in cursor.fetchall():
                text_store[i_id] = np.frombuffer(vec_blob, dtype=np.float32)
            self.text_index.vector_store = text_store
            
            conn.close()
            print("test msg : Loaded local vector DB from SQLite successfully.")
            return True
        except Exception as e:
            print(f"Error loading local vector DB: {e}")
            return False

# Global instance of the local vector database
vector_db = LocalVectorDB()
vector_db.load()
