# LostLink AI
## High-Performance, Standalone Multi-Modal RAG Platform & Decoupled Vector Search Engine

LostLink AI is a production-grade, self-contained intelligent asset recovery platform. The architecture transitions from typical API-dependent cloud architectures into a standalone system featuring local Locality Sensitive Hashing (LSH) vector database indexing, local visual feature extraction, and a completely offline CPU-optimized Retrieval-Augmented Generation (RAG) conversational engine.

The core philosophy:
> **Deterministic scoring first. AI reasoning as reinforcement - not replacement.**

Every match is backed by measurable similarity metrics, ensuring explainability, auditability, and production-grade reliability.

---

## 1. System Architecture

```mermaid
flowchart TD
    subgraph Client Layer [Client Interface]
        UI[Vanilla JS / HTML5 Frontend]
        Chat[RAG Chat Copilot]
    end

    subgraph API Gateway [FastAPI Gateway]
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

## 2. Core Engineering Architecture

### Multi-Modal Matching Engine (M3E)
Each report passes through a layered, multi-modal verification system:

#### Layer 1: Visual Feature Extraction (Local Inference)
* **Engine**: MobileNetV2 (PyTorch / torchvision)
* **Embedding Size**: 1024–1280 dimensions
* **Similarity Metric**: Cosine Similarity
* **Activation Threshold**: 0.82 (configurable)

##### Preprocessing Pipeline
Before inference, images undergo standardized preparation:
* Resize to 256px
* Center-crop to 224px
* RGB conversion
* Tensor normalization (using ImageNet means: `[0.485, 0.456, 0.406]` and std: `[0.229, 0.224, 0.225]`)

This guarantees consistent feature maps regardless of original resolution and acts as a neural compression layer.

##### Why Local Inference?
* Eliminates cloud latency for visual matching
* Enables sub-second repository-wide vector comparisons
* Reduces API dependency and cost

#### Layer 2: Lexical & Contextual Matching (Stateless Feature Hashing)
* **Engine**: Scikit-Learn `HashingVectorizer`
* **Technique**: Character and Word-level Trigram Analysis (`ngram_range=(1,3)`)
* **Representation**: Sparse matrix ($2^{10}$ / 1024-dimensional space)

The system captures semantic substructures such as:
* "black backpack"
* "broken hinge"
* "library near LHC"

By replacing standard `TfidfVectorizer` with `HashingVectorizer`, the vectorization is completely stateless. This ensures consistent vector mappings across multiple nodes without requiring shared vocabulary dictionaries.

#### Layer 3: Spatio-Temporal Correlation Engine
##### Spatial Logic
A landmark-aware matching algorithm weights results based on proximity to 50+ predefined campus landmarks (NITK Surathkal optimized), including:
* Hostels (GH-1 to GH-6, Mega Towers)
* LHC-A, LHC-B
* Central Library
* Main Building
* SAC, SJA
* Sports Complex
* NITK Beach

Matches sharing validated landmark tags receive weighted boosts.

##### Temporal Validation
A sequence-validation algorithm enforces:
$$T_{\text{lost}} < T_{\text{found}}$$
This prevents false positives from historical records and improves reliability of match suggestions.

---

## 3. Decoupled Local Vector Database Design

To ensure horizontal scaling across multi-node deployments without external dictionary synchronization, LostLink implements a **custom Locality Sensitive Hashing (LSH) index** for sub-linear semantic retrieval.

### Locality Sensitive Hashing (LSH) Mechanics
High-dimensional dense features are mapped to low-dimensional binary keys using random projection. Given a vector $v$ and a random projection matrix $R \in \mathbb{R}^{K \times D}$:
$$h(v) = \text{sign}(R \cdot v)$$
This results in a $K$-bit binary hash representing the signature bucket.
* **Bucket Indexing**: Buckets are persisted locally using a serialized Python `.pkl` cache store, synchronized automatically on application boot from the MongoDB document store.
* **Search Speed**: Finding nearest neighbors is reduced to calculating the Hamming Distance between the query signature and indexed keys, resulting in $O(\log N)$ sub-linear retrieval complexity compared to $O(N)$ database scans.

---

## 4. Conversational RAG Architecture

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

## 5. Image Engineering & Payload Optimization

High-resolution image uploads can cause increased payload sizes, slower mobile uploads, higher storage usage, and memory spikes during inference. LostLink implements a two-stage optimization pipeline:

### Stage 1: Intelligent Image Compression
Before storage:
* Images are resized
* Quality is reduced within perceptual tolerance
* RGB standardization is applied

This reduces upload latency, storage footprint, and inference memory load.

### Stage 2: Image → Semantic Text Transformation
Instead of repeatedly transmitting image binaries:
1. Image is analyzed once.
2. Semantic description is generated.
3. Structured textual representation is stored.

This allows search matchers to perform similarity evaluations on image-derived text, reducing repeated cloud API calls and generating searchable metadata.

---

## 6. Recovery Infrastructure & Features

### QR-Based Offline-to-Online Bridge
* Unique QR codes are generated per item using the `qrcode` library and encoded in Base64.
* Printable tags can be affixed to physical assets.
* Scanning the QR opens a secure recovery portal where the finder can notify the owner without needing an account.

### Multi-Channel Notification Pipeline
* **SMTP Email Alerts**: Automated upon high-confidence matches, containing direct claim links and context summaries.
* **Voice Engine (Local)**: Local `pyttsx3`-based speech alerts.
* **Telephony Integration**: Twilio API integration to trigger automated phone calls for high-priority reports.

---

## 7. Authentication & Security

### JWT Infrastructure
* Stateless session management.
* Bearer token authentication.
* Protected API endpoints.

### Password Security
* `bcrypt` salted hashing.
* 72-byte safe validation.
* Secure credential storage.

---

## 8. Performance Benchmarks

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

## 9. Research Positioning: Deterministic, Not Agentic

LostLink AI is a deterministic trigger-response infrastructure where every match is backed by:
* Cosine Similarity scores
* HashingVectorizer weight distributions
* Spatio-temporal validation
* Configurable thresholds

AI/LLM is used purely as a verification and formatting layer, not a decision-making authority. This design guarantees **explainability**, **auditability**, **predictability**, and **production reliability**.

---

## 10. Demonstrated Engineering Competencies

* Multi-modal AI system design
* Neural feature extraction pipelines
* NLP similarity modeling (HashingVectorizer & Trigrams)
* Local CPU LLM integration with direct AutoModel classes
* High-performance REST API design
* Async backend engineering (FastAPI BackgroundTasks)
* Image compression optimization
* Security-first authentication systems
* Containerized multi-container cloud deployment

---

## 11. Docker & Local Deployment Setup

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
