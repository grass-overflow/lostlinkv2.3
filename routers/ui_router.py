from fastapi import APIRouter
from fastapi.responses import FileResponse
import os

router = APIRouter(tags=["UI"])

FRONTEND_DIR = "frontend"

@router.get("/")
@router.get("/index.html")
def serve_home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@router.get("/login")
@router.get("/login.html")
def serve_login():
    return FileResponse(os.path.join(FRONTEND_DIR, "login.html"))

@router.get("/signup")
@router.get("/signup.html")
def serve_signup():
    return FileResponse(os.path.join(FRONTEND_DIR, "signup.html"))

@router.get("/dashboard")
def serve_dashboard():
    return FileResponse(os.path.join(FRONTEND_DIR, "dashboard.html"))

@router.get("/report_lost")
@router.get("/report_lost.html")
def serve_report_lost():
    return FileResponse(os.path.join(FRONTEND_DIR, "report_lost.html"))

@router.get("/report_found")
@router.get("/report_found.html")
def serve_report_found():
    return FileResponse(os.path.join(FRONTEND_DIR, "report_found.html"))

@router.get("/browser")
def serve_browser():
    return FileResponse(os.path.join(FRONTEND_DIR, "browser.html"))

@router.get("/feedback_page")
def serve_feedback():
    return FileResponse(os.path.join(FRONTEND_DIR, "feedback.html"))

@router.get("/whats_new")
def serve_whats_new():
    return FileResponse(os.path.join(FRONTEND_DIR, "whats_new.html"))

@router.get("/qr/{item_id}")
def serve_qr_page(item_id: str):
    return FileResponse(os.path.join(FRONTEND_DIR, "qr_page.html"))

@router.get("/admin")
def serve_admin_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "admin.html"))

@router.get("/premium")
def serve_premium_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "premium.html"))
