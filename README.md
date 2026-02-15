# LostLink AI  
## Robust Multi-Modal Matching Infrastructure for Intelligent Asset Recovery

LostLink AI is a recovery system designed to solve the Lost & Found problem through a deterministic, research-driven matching pipeline.

Unlike conventional keyword-based systems, LostLink integrates local computer vision, structured NLP similarity, spatio-temporal validation, and optional LLM verification into a transparent and auditable decision engine.


---

# 1. System Overview

LostLink is architected as a high-performance FastAPI backend with a MongoDB document store, containerized for scalable deployment.

The core philosophy:

> Deterministic scoring first.  
> AI reasoning as reinforcement - not replacement.

Every match is backed by measurable similarity metrics, ensuring explainability and reliability.

---

# 2. Core Engineering Architecture

## Multi-Modal Matching Engine (M3E)

Each report passes through a layered verification system:

---

## Layer 1: Visual Feature Extraction (Local Inference)

**Engine:** MobileNetV2 (PyTorch / torchvision)  
**Embedding Size:** 1024–1280 dimensions  
**Similarity Metric:** Cosine Similarity  
**Activation Threshold:** 0.82 (configurable)

### Neural Preprocessing Pipeline

Before inference, images undergo standardized preparation:

- Resize to 256px
- Center-crop to 224px
- RGB conversion
- Tensor normalization

This guarantees consistent feature maps regardless of original resolution and acts as a neural compression layer.

### Why Local Inference?

- Eliminates cloud latency for visual matching  
- Enables sub-second repository-wide vector comparisons  
- Reduces API dependency and cost  

---

## Layer 2: Lexical & Contextual Matching

**Engine:** Scikit-Learn `TfidfVectorizer`  
**Technique:** Trigram Analysis (`ngram_range=(1,3)`)  
**Representation:** Sparse matrix  

The system captures semantic substructures such as:

- “black backpack”
- “broken hinge”
- “library near LHC”

Trigram analysis significantly improves contextual matching compared to unigram-only systems.

Sparse matrix optimization ensures repository-wide similarity scans execute in sub-second time.

---

## Layer 3: Spatio-Temporal Correlation Engine

### Spatial Logic

A landmark-aware matching algorithm weights results based on proximity to 50+ predefined campus landmarks (NITK Surathkal optimized), including:

- Hostels (GH-1 to GH-6, Mega Towers)
- LHC-A, LHC-B
- Central Library
- Main Building
- SAC, SJA
- Sports Complex
- NITK Beach

Matches sharing validated landmark tags receive weighted boosts.

### Temporal Validation

A sequence-validation algorithm enforces:

```

Lost Timestamp < Found Timestamp

```

This prevents false positives from historical records and improves reliability of match suggestions.

---

## Layer 4: AI Reasoning Fallback (Premium Verification)

**Provider:** Google Gemini Flash  

Role:

- High-precision semantic cross-verification  
- Context nuance detection  
- Description refinement  

LLM reasoning is used as a reinforcement layer, not the primary matching driver.

This ensures:

- Deterministic transparency  
- Reduced hallucination risk  
- Controlled cost exposure  

---

# 3. Image Engineering & Payload Optimization

High-resolution image uploads can cause:

- Increased payload sizes  
- Slower mobile uploads  
- Higher storage usage  
- Memory spikes during inference  

LostLink implements a two-stage optimization pipeline.

---

## Stage 1: Intelligent Image Compression

Before storage:

- Images are resized
- Quality is reduced within perceptual tolerance
- RGB standardization is applied

Impact:

- Reduced upload latency  
- Lower storage footprint  
- Reduced inference memory load  

---

## Stage 2: Image → Semantic Text Transformation

Instead of repeatedly transmitting image binaries:

1. Image analyzed once
2. Semantic description generated
3. Structured textual representation stored

This allows:

- TF-IDF matching on image-derived text  
- Efficient cross-report comparison  
- Reduced repeated cloud API calls  
- Searchable structured metadata  

The system effectively converts visual information into reusable semantic assets.

---

# 4. Recovery Infrastructure

## QR-Based Offline-to-Online Bridge

- Unique QR codes generated per item
- Encoded via `qrcode` and Base64
- Printable tags for physical assets

Scanning the QR:

- Opens secure recovery portal
- Allows finder to notify owner
- No account required for finder

---

## Multi-Channel Notification Pipeline

### SMTP Email Alerts
- Automated upon high-confidence match
- Direct claim links
- Context summary included

### Voice Engine (Local)
- pyttsx3-based speech alerts

### Telephony Integration
- Twilio API for automated phone calls
- Triggered for high-priority reports

---

# 5. Authentication & Security

## JWT Infrastructure

- Stateless session management
- Bearer token authentication
- Protected routes

## Password Security

- bcrypt salted hashing
- 72-byte safe validation
- Secure credential storage

---

# 6. Performance Engineering & Optimizations

## Lazy Model Loading

Problem:
Loading MobileNet during server startup caused high memory allocation.

Solution:
Singleton-based lazy initialization — model loads only on first inference event.

Impact:
- ~80% reduction in cold-start latency
- Reduced idle RAM usage

---

## Asynchronous Notification Tasks

Problem:
Twilio and SMTP calls blocked request cycle.

Solution:
FastAPI BackgroundTasks for non-blocking execution.

Impact:
- Instant user confirmation
- Improved throughput
- Better perceived performance

---

## Sparse Matrix Efficiency

TF-IDF vectors stored as sparse matrices:

- Efficient memory usage
- Fast similarity sweep across large repositories
- Scalable to thousands of items

---

# 7. System Architecture Flow

1. User uploads image + metadata  
2. FastAPI validates schema  
3. Image preprocessing & compression  
4. MobileNet extracts embedding  
5. TF-IDF vectorization of text  
6. Spatio-temporal scoring applied  
7. Weighted ensemble aggregation  
8. If score > threshold → Trigger notification pipeline  
9. Admin dashboard logs match for audit  
10. Secure claim initiated  

---

# 8. Deployment & Infrastructure

## Backend
- Python
- FastAPI
- Uvicorn

## Database
- MongoDB
- Flexible schema for diverse item metadata

## ML Stack
- PyTorch
- Scikit-Learn
- Google Generative AI (Gemini)

## Frontend
- Vanilla JavaScript
- Modern CSS (Glassmorphism + Shimmer effects)

## DevOps
- Docker (multi-stage build)
- Render deployment
- Persistent disk storage for uploads
- Twilio API
- SMTP mail services

---

# 9. Research Positioning: Deterministic, Not Agentic

LostLink AI is not an autonomous agent.

It is a deterministic trigger-response infrastructure where every match is backed by:

- Cosine Similarity scores  
- TF-IDF weight distributions  
- Spatio-temporal validation  
- Configurable thresholds  

AI is used as verification, not decision-making authority.

This ensures:

- Explainability  
- Auditability  
- Predictability  
- Production reliability  

---

# 10. Demonstrated Engineering Competencies

- Multi-modal AI system design  
- Neural feature extraction pipelines  
- NLP similarity modeling  
- LLM integration with controlled fallback  
- High-performance REST API design  
- Async backend engineering  
- Image compression optimization  
- Security-first authentication systems  
- Containerized cloud deployment  

---

## Contact

Rohith  
rohith02aug@gmail.com  

---

LostLink AI is a scalable, explainable, and performance-optimized recovery infrastructure engineered with production intent.
```

