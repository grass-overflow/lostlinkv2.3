import time
import numpy as np
from vector_db import LSHVectorIndex

def run_retrieval_benchmark(num_items=10000):
    print(f"--- Running Retrieval Benchmark (Scale: N = {num_items} items) ---")
    
    # 1. Initialize indices
    image_dim = 1280
    text_dim = 128
    
    image_index = LSHVectorIndex(image_dim, num_hyperplanes=12, seed=42)
    text_index = LSHVectorIndex(text_dim, num_hyperplanes=8, seed=101)
    
    # 2. Generate random mock items
    print("Generating mock vector database...")
    np.random.seed(42)
    mock_images = np.random.randn(num_items, image_dim)
    mock_texts = np.random.randn(num_items, text_dim)
    
    print("Inserting items into LSH index...")
    for i in range(num_items):
        item_id = f"item_{i}"
        image_index.insert(item_id, mock_images[i])
        text_index.insert(item_id, mock_texts[i])
        
    # Generate query vectors
    query_image = np.random.randn(image_dim)
    query_text = np.random.randn(text_dim)
    
    # 3. Benchmark Linear Lookup (Brute-Force Cosine Similarity)
    print("Benchmarking Linear Scan...")
    start_time = time.perf_counter()
    
    # Brute-force linear scan calculation
    linear_results = []
    norm_q = np.linalg.norm(query_text)
    if norm_q > 0:
        for item_id, cand_v in text_index.vector_store.items():
            norm_c = np.linalg.norm(cand_v)
            if norm_c > 0:
                sim = np.dot(query_text, cand_v) / (norm_q * norm_c)
            else:
                sim = 0.0
            linear_results.append((item_id, float(sim)))
    linear_results.sort(key=lambda x: x[1], reverse=True)
    
    linear_duration = (time.perf_counter() - start_time) * 1000 # in ms
    
    # 4. Benchmark LSH Query Lookup
    print("Benchmarking LSH Bucket Lookup...")
    start_time = time.perf_counter()
    lsh_results = text_index.query(query_text, max_hamming_distance=3)
    lsh_duration = (time.perf_counter() - start_time) * 1000 # in ms
    
    # 5. Output Comparison Table
    speedup = linear_duration / lsh_duration if lsh_duration > 0 else 0
    print("\n" + "="*50)
    print(" RETRIEVAL BENCHMARKS RESULT ")
    print("="*50)
    print(f"Database Scale:       {num_items} items")
    print(f"Linear Scan Latency:  {linear_duration:.2f} ms")
    print(f"LSH Query Latency:    {lsh_duration:.2f} ms")
    print(f"Speedup Ratio:        {speedup:.1f}x Faster")
    print(f"Result Alignment:     {len(lsh_results)} candidates retrieved out of {num_items}")
    print("="*50 + "\n")

def run_llm_benchmark():
    print("--- Running Local LLM Generation Benchmark ---")
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        
        print("Loading local FLAN-T5-Small model (google/flan-t5-small)...")
        start_load = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        load_duration = time.perf_counter() - start_load
        print(f"Model Loaded in: {load_duration:.2f} seconds")
        
        # Test input
        prompt = (
            "Answer the user query based on the following context.\n\n"
            "Context: Item 1: Lenovo Ideapad found at Central Library. Description: Grey laptop. Claim ID is 6a1bf11b496a. "
            "User Query: Did anyone find my grey Lenovo laptop?\n\n"
            "Answer:"
        )
        
        print("Generating answer (running CPU inference)...")
        start_infer = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        infer_duration = (time.perf_counter() - start_infer) * 1000
        
        print("\n" + "="*50)
        print(" LOCAL LLM BENCHMARKS RESULT ")
        print("="*50)
        print(f"Model ID:             google/flan-t5-small (80M Parameters)")
        print(f"Inference Latency:    {infer_duration:.2f} ms")
        print(f"Generated Output:     \"{response}\"")
        print("="*50 + "\n")
        
    except ImportError:
        print("\nNote: 'transformers' or 'torch' not found in this environment.")
        print("To run the local LLM benchmark, install packages or execute this script inside the running Docker container:")
        print("`docker compose exec lostlink-api python3 benchmark.py`\n")

if __name__ == "__main__":
    run_retrieval_benchmark(num_items=10000)
    run_llm_benchmark()
