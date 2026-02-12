# LostLink: AI-Powered Campus Asset Recovery

LostLink is a multi-modal matching engine designed to solve the lost-and-found problem within large-scale campus environments (specifically optimized for NITK Surathkal). It leverages a hybrid approach combining local Computer Vision (CV) and Large Language Models (LLMs) to automate item discovery.

## 🚀 Engineering Highlights

### 1. Hybrid Multi-Modal Matching Engine
The core of LostLink is an ensemble matching system that processes reports through three distinct logical layers:

*   **Visual Similarity Layer (Local ML):**
    *   Utilizes **MobileNetV2** for feature extraction.
    *   Generates a 1280-dimensional feature vector (embedding) for every reported item.
    *   Calculates **Cosine Similarity** between new reports and existing records to identify visual matches with low latency.
*   **Semantic Reasoner (LLM Layer):**
    *   Integrates **Google Gemini 1.5 Flash** for deep textual and image analysis.
    *   Analyzes unstructured descriptions to confirm matches where visual data might be ambiguous (e.g., "damaged corner," "specific brand logo").
*   **Spatio-Temporal Logic (NLP Layer):**
    *   Implements **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization with n-gram analysis.
    *   **Campus-Aware Location Scoring:** Includes a dictionary of 50+ NITK-specific locations (Hostels, Lecture Halls, Facilities).
    *   **Logic:** Matches receive a 30% scoring boost if they share a recognized campus landmark and a 15% boost if the "Lost" timestamp logically precedes the "Found" timestamp.

### 2. High-Performance Backend Architecture
*   **Asynchronous Processing:** Built on **FastAPI** to handle concurrent I/O-bound tasks and high-frequency report submissions.
*   **Flexible Schema:** Utilizes **MongoDB** for storage, allowing for varied item metadata without the overhead of complex relational migrations.
*   **Lazy Loading:** ML models are loaded into memory on-demand to optimize resource utilization (crucial for lightweight deployment environments).

### 3. Security and Authentication
*   **Bcrypt-SHA256 Hashing:** Implements a pre-hashing strategy to bypass the standard 72-character limit of Bcrypt, supporting long, secure passphrases.
*   **JWT Authorization:** Stateless session management using securely signed JSON Web Tokens.

### 4. Admin Observability & Audit
*   **Match Transparency Dashboard:** A dedicated interface for administrators to audit the AI's "decision-making" process.
*   **Match Logs:** Every match is recorded with its logic type (Visual vs. Semantic vs. Text), confidence score, and specific reasoning string.

## 🗺️ Localized Knowledge Base
The system is pre-configured with the geography of **NITK Surathkal**, including:
*   **Residential:** All 20+ Hostel Blocks (Mega Towers 1-3, Karavali, GH-1 to GH-6, etc.).
*   **Academic:** LHC-A, LHC-B, Main Building, Central Library, CCC.
*   **Facilities:** SJA, SAC, STEP, Sports Complexes, and NITK Beach.

## 🛠️ Technology Stack
*   **Language:** Python 3.9+
*   **Framework:** FastAPI
*   **Database:** MongoDB
*   **ML/AI:** PyTorch (MobileNetV2), Google Generative AI (Gemini), Scikit-Learn (TF-IDF/Cosine Similarity)
*   **Infrastructure:** Docker, Render (Auto-deployment)
