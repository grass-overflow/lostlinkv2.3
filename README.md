# 🔍 LostLink AI: Intelligent Asset Recovery System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-Database-47A248?logo=mongodb&logoColor=white)](https://www.mongodb.com)
[![AI/ML](https://img.shields.io/badge/AI%20Matching-Gemini%20%2B%20MobileNetV2-blueviolet?logo=google-gemini&logoColor=white)]()

**LostLink AI** is a state-of-the-art, full-stack Lost and Found ecosystem designed to bridge the gap between finders and owners using Machine Learning and automated notification agents.

---

## 🚀 Key Features

### 🧠 Hybrid AI Matching Engine
Our proprietary matching logic combines two engineering layers to ensure high accuracy even with sparse data:
*   **Visual Vector Similarity**: Uses a locally hosted **MobileNetV2** (PyTorch) to extract 1280-dimensional feature embeddings from item photos. Matches are calculated using **Cosine Similarity** scores.
*   **Semantic Text Matching**: Implements an **N-gram (1-3) TF-IDF Vectorizer** that analyzes descriptions and locations. It prioritizes logical flow (e.g., *Lost Date < Found Date*).
*   **Gemini Reasoning**: Integrates **Google Gemini 1.5 Flash/Pro** for semantic image description and high-level cross-referencing between reports.

### 🏷️ Offline-to-Online Recovery (QR Shield)
*   **Printable Physical Tags**: Users can generate and print high-resolution QR tags to attach to keys, laptops, or wallets.
*   **Recovery Shield Page**: Scanning a lost item's QR code opens a secure recovery portal where finders can instantly notify the owner with one click—no account required for finders.

### 🤖 Multi-Channel Notification Agent
*   **AI Calling**: Automated phone calls via **Twilio Voice API** to alert users the moment a high-priority match is found.
*   **Smart Email Alerts**: Context-aware email notifications with direct links to the matching reports.
*   **Voice Synthesis**: Local text-to-speech (pyttsx3) for developer-side system status and alerts.

### 🛡️ Secure Infrastructure
*   **Admin Control Center**: A centralized dashboard for community moderation, featuring full CRUD on reports and feedback analytics.
*   **JWT Authentication**: Secure user sessions with encrypted token storage.
*   **Rate Limiting**: Anti-spam protection with daily reporting limits to maintain database integrity.

---

## 🛠️ Tech Stack & Engineering Decisions

| Layer | Technology | Decision Rationale |
| :--- | :--- | :--- |
| **Backend** | FastAPI | Asynchronous performance and automatic Swagger/OpenAPI documentation. |
| **Database** | MongoDB Atlas | Flexible schema for diverse item attributes and high availability. |
| **ML Inference** | PyTorch (MobileNetV2) | **Lazy Loading Implementation**: The model is initialized only on the first scan, reducing cold-boot RAM usage significantly. |
| **Generative AI**| Gemini API | Leveraged for advanced image-to-text conversion to handle noisy user descriptions. |
| **Frontend** | Vanilla JS / CSS | Zero-dependency frontend optimized for speed and maximum control over the **Glassmorphism Design System**. |
| **Deployment** | Environment Driven | Uses `BASE_URL` abstraction to handle seamless transitions between local development and production URLs. |
| **Containerization** | Docker | Containerized using Multi-stage Docker builds to ensure environment consistency. |

---

## 🛠️ DevOps & Scaling Strategy
To demonstrate production-grade DevOps skills, the project follows these patterns:

*   **Dockerized Environment**: The app is containerized via `Dockerfile` and `docker-compose.yml`, including health checks and persistent volume mounting for item images.
*   **Horizontal Scaling Ready**: The FastAPI backend is stateless (JWT-based), allowing for easy horizontal scaling behind an Nginx or Traefik load balancer.
*   **Database Scalability**: By using **MongoDB Atlas**, we leverage built-in sharding and replica sets for high availability.
*   **Static Asset Optimization**: The frontend is served directly through FastAPI's `StaticFiles`, but can be offloaded to a CDN (CloudFront/Cloudflare) or Nginx for improved edge performance.

---

### 🚀 Deployment Options

#### Option A: Run with Docker Compose (Local/VPS)
1.  **Configure `.env`**: Add your keys.
2.  **Run**:
    ```bash
    docker-compose up --build -d
    ```

#### Option B: Deploy to Render (Cloud)
1.  **Push code to GitHub**.
2.  **Log in to Render** and click "New" > "Blueprint".
3.  **Connect your Repo**: It will automatically detect `render.yaml`.
4.  **Fill Environment Variables**: Copy the keys from your `.env` into the Render dashboard.
5.  **Set BASE_URL**: Set it to your Render assigned URL (e.g., `https://your-app.onrender.com`).

---

## 🎨 Design Philosophy
The UI is built on a **Premium Glassmorphism** aesthetic using the **Outfit** font family. 
*   **Accessibility**: High-contrast states for lost (Rose) vs found (Emerald) items.
*   **Responsiveness**: Mobile-first architecture ensures that finders in the field can report items easily on the go.
*   **UX Micro-interactions**: Smooth transitions, loading states, and AI "scanning" animations for a professional feel.

---

## 📈 System Workflow (Lifecycle)
1.  **Report**: User uploads item details + photo.
2.  **Feature Extraction**: AI Agent extracts semantic text features and visual embeddings.
3.  **Cross-Reference**: System queries MongoDB using vectorized similarity searches.
4.  **Verification**: If similarity > 0.82, the AI triggers the notification pipeline.
5.  **Recovery**: Finder scans physical QR tag → Owner gets a phone call/email → Secure claim initiated.

---

## 👨‍💻 Developer Information
Built as a high-fidelity resume project to demonstrate proficiency in:
*   Full-stack Architecture
*   ML Integration (Computer Vision & NLP)
*   API Design & Security
*   Cloud Database Management

**Contact for Collaboration:** [rohith02aug@gmail.com](mailto:rohith02aug@gmail.com)
