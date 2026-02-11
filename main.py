import os
from dotenv import load_dotenv

# Load environment variables before importing other modules
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import routers
from routers.auth import router as auth_router
from routers.items_router import router as items_router
from routers.admin_router import router as admin_router
from routers.feedback_router import router as feedback_router
from routers.ui_router import router as ui_router

app = FastAPI(title="LostLink AI API")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
os.makedirs("uploads", exist_ok=True)

# 1. Mount uploads (higher priority than root mount)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# 2. Include API Routers (routes take precedence over final static mount)
app.include_router(auth_router, prefix="/api")
app.include_router(items_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
app.include_router(feedback_router, prefix="/api")
app.include_router(ui_router)

# 3. Final Fallback: Mount frontend directory at / for assets (style.css, img/, etc.)
# This ensures that if a URL doesn't match a route, FastAPI looks in the frontend folder.
app.mount("/", StaticFiles(directory="frontend"), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
