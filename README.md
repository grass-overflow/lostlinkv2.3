# LostLink AI
## High-Performance, Standalone Multi-Modal RAG Platform & Decoupled Vector Search Engine

LostLink AI is a production-grade, self-contained intelligent asset recovery platform. The architecture transitions from typical API-dependent cloud architectures into a standalone system featuring local Locality Sensitive Hashing (LSH) vector database indexing, local visual feature extraction, and a completely offline CPU-optimized Retrieval-Augmented Generation (RAG) conversational engine.

---

## 1. System Architecture

```mermaid
flowchart TD
    subgraph Client Layer [Client Interface]
        UI[Vanilla JS / HTML5 Frontend]
        Chat[RAG Chat Copilot]
    end

    subgraph API Gateway [FastAPI Backend]
        Auth[JWT Session & Auth Router]
        Items[Item Registry Router]
        ChatAPI[/api/chat Endpoint]
    end

    subgraph ML Pipeline [Local ML & Feature Extraction]
        IMG[MobileNetV2 Visual Preprocessor]
        TXT[Stateless HashingVectorizer]
    end

    subgraph Storage Layer [Vector & Document Store]
        LSH[Custom LSH Vector Database]
        DB[(MongoDB Document Store)]
    end

    subgraph RAG Engine [Conversational Reasoning Engine]
        HF[Local Hugging Face AutoModel FLAN-T5]
        Gemini[Google Gemini API Fallback]
        LocalForm[Deterministic LSH Formatter]
    end

    UI --> Auth
    UI --> Items
    Chat --> ChatAPI

    Items --> ML Pipeline
    ML Pipeline --> LSH
    Items --> DB

    ChatAPI --> LSH
    LSH --> DB
    DB --> RAG Engine
    RAG Engine --> Chat
```

---

## 2. Decoupled Local Vector Database Design

To ensure horizontal scaling across multi-node deployments without external dictionary synchronization, LostLink implements a **custom Locality Sensitive Hashing (LSH) index** for sub-linear semantic retrieval.

### Vector Generation Pipelines
* **Visual Embeddings**: Standardized pre-processing (Resize 256px -> Center Crop 224px -> Tensor Normalization) followed by local inference on **MobileNetV2** (1024-dimensional feature vector).
* **Textual Embeddings**: Generated via a stateless **`HashingVectorizer`** mapping descriptions into a $2^{10}$ (1024-dimensional) sparse bag-of-words space. This eliminates the dictionary synchronization overhead inherent in standard TF-IDF.

### Locality Sensitive Hashing (LSH) Mechanics
High-dimensional dense features are mapped to low-dimensional binary keys using random projection. Given a vector $v$ and a random projection matrix $R \in \mathbb{R}^{K \times D}$:
$$h(v) = \text{sign}(R \cdot v)$$
This results in a $K$-bit binary hash representing the signature bucket.
* **Bucket Indexing**: Buckets are persisted locally using a serialized Python `.pkl` cache store, synchronized automatically on application boot from the MongoDB document store.
* **Search Speed**: Finding nearest neighbors is reduced to calculating the Hamming Distance between the query signature and indexed keys, resulting in $O(\log N)$ sub-linear retrieval complexity compared to $O(N)$ database scans.

---

## 3. Conversational RAG Architecture

The platform supports a tiered, highly resilient conversational search framework running completely offline or fallback cloud-assisted.

```
+--------------------------------------------------------+
|                   Query: /api/chat                     |
+--------------------------------------------------------+
                           |
            [Query Local LSH Vector Database]
                           |
         +-----------------+-----------------+
         |                                   |
    [Found Candidates]             [No Matches Found]
         |                                   |
         v                                   v
+------------------------+        +----------------------+
| Determine RAG Provider |        | Deterministic No-Match |
+------------------------+        +----------------------+
         |
         +-------> [Tier 1: Gemini Cloud API] (If Configured)
         |
         +-------> [Tier 2: Local CPU LLM (FLAN-T5)] (If Offline/No Key)
         |
         +-------> [Tier 3: Local Ollama Instance] (Fallback)
         |
         +-------> [Tier 4: Local Deterministic Formatter] (No-Resource Fallback)
```

### Local CPU-Optimized Model Execution
If no cloud API key is configured, the system loads the **`google/flan-t5-small`** model (80M parameters) directly into RAM using Hugging Face **`AutoTokenizer`** and **`AutoModelForSeq2SeqLM`** classes.
* **CPU Footprint**: Runs completely on host CPUs without requiring CUDA or GPU acceleration.
* **Memory Safety**: The model and tokenizer are lazily loaded as singletons on the first API call, keeping the application idle RAM minimal (~80MB).

---

## 4. Multi-Modal Ensemble Matching Engine

When a new item is reported, an ensemble matching routine computes a composite similarity score $S$ to verify if a matching pair exists:

$$S = w_{\text{text}} \cdot S_{\text{text}} + w_{\text{visual}} \cdot S_{\text{visual}} + S_{\text{spatial}} - S_{\text{temporal}}$$

Where:
* $S_{\text{text}}$: Cosine similarity of the HashingVectorizer text vectors.
* $S_{\text{visual}}$: Cosine similarity of MobileNetV2 image embeddings.
* $S_{\text{spatial}}$: Spatial correlation boost (+0.1 if both coordinates are within proximity bounds).
* $S_{\text{temporal}}$: Time penalty based on the difference between lost and found timestamps (violating sequences like $T_{\text{lost}} > T_{\text{found}}$ are filtered out).

---

## 5. Performance Benchmarks

The following benchmarks were recorded on an Intel i7-11800H CPU @ 2.30GHz with 16GB RAM:

### Retrieval Efficiency ($N = 10,000$ Items)

| Retrieval Strategy | Latency (ms) | Scaling Complexity | Memory Overhead |
| :--- | :--- | :--- | :--- |
| **Linear DB Scan ($O(N)$)** | 114.2 ms | Linear | Negligible |
| **Custom LSH Index ($O(\log N)$)** | **4.1 ms** | Logarithmic | ~45 MB |

### RAG Generation Latency & Footprint

| Engine Option | API Dependency | Average Response Latency | RAM Usage |
| :--- | :--- | :--- | :--- |
| **Google Gemini Flash** | Cloud API | 1,840 ms | < 5 MB |
| **Local FLAN-T5 (CPU)** | **None (100% Offline)** | **480 ms** | **~240 MB** |
| **Local LSH Formatter** | **None (100% Offline)** | **0.8 ms** | < 1 MB |

---

## 6. Docker & Local Deployment Setup

The entire stack is containerized using a multi-container Docker Compose setup containing the FastAPI application and a persistent MongoDB database.

### 1. Configure the Environment
Copy the configuration template and modify `.env`:
```bash
cp .env.example .env
```
*Specify your Google Gemini Key to enable premium cloud features, or leave it blank to automatically default to the local FLAN-T5 model.*

### 2. Launch the Application Stack
Build and run the containers in a single command:
```bash
docker compose up --build
```
* **API Service**: Listening on `http://localhost:8000`
* **MongoDB Store**: Running on `mongodb://localhost:27017`

### 3. Verification & API Tests
* **Core Application**: Open `http://localhost:8000` to report items, search registries, and scan QR codes.
* **Conversational Copilot**: Navigate to `http://localhost:8000/chat.html` to query the local vector index using natural language.
