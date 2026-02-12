import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


from routers.auth import router as auth_router
from routers.items_router import router as items_router
from routers.admin_router import router as admin_router
from routers.feedback_router import router as feedback_router
from routers.ui_router import router as ui_router

app = FastAPI(title="LostLink AI API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "online", "timestamp": str(__import__("datetime").datetime.now())}


os.makedirs("uploads", exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.include_router(auth_router, prefix="/api")
app.include_router(items_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
app.include_router(feedback_router, prefix="/api")
app.include_router(ui_router)
app.mount("/", StaticFiles(directory="frontend"), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
