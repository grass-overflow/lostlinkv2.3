from fastapi import APIRouter, Form
from datetime import datetime
from database import feedback_col

router = APIRouter(tags=["feedback"])

@router.post("/feedback")
def submit_feedback(name: str = Form(...), email: str = Form(None), message: str = Form(...)):
    feedback = {
        "name": name,
        "email": email,
        "message": message,
        "date": datetime.utcnow()
    }
    feedback_col.insert_one(feedback)
    return {"message": "Thank you for your feedback!"}

@router.get("/feedbacks")
def get_feedbacks():
    feedbacks = []
    for fb in feedback_col.find().sort("date", -1):
        feedbacks.append({
            "id": str(fb.get("_id")),
            "name": fb.get("name", "Anonymous"),
            "email": fb.get("email"),
            "message": fb.get("message"),
            "date": fb.get("date").strftime("%Y-%m-%d %H:%M:%S") if fb.get("date") else "Unknown"
        })
    return feedbacks
