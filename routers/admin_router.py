import os
from fastapi import APIRouter, HTTPException, Depends
from bson import ObjectId
from database import items_col, feedback_col, users_col
from routers.auth import get_current_user

router = APIRouter(prefix="/admin", tags=["admin"])

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "rohith02aug@gmail.com").strip().lower()
if ADMIN_EMAIL.startswith('"') and ADMIN_EMAIL.endswith('"'):
    ADMIN_EMAIL = ADMIN_EMAIL[1:-1]

def admin_only(user: dict = Depends(get_current_user)):
    user_email = user.get("email", "").strip().lower()
    if user_email != ADMIN_EMAIL:
        print(f"🚫 Access Denied for: {user_email} (Expected: {ADMIN_EMAIL})")
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

def sanitize_item(item):
    item["_id"] = str(item["_id"])
    return item

@router.get("/is_admin")
def check_admin(user: dict = Depends(get_current_user)):
    user_email = user.get("email", "").strip().lower()
    return {"is_admin": user_email == ADMIN_EMAIL}

@router.get("/items")
def get_admin_items(user: dict = Depends(admin_only)):
    return [sanitize_item(item) for item in items_col.find()]

@router.get("/feedbacks")
def get_feedbacks(user: dict = Depends(admin_only)):
    feedbacks = []
    for fb in feedback_col.find().sort("date", -1):
        feedbacks.append({
            "_id": str(fb["_id"]),
            "name": fb.get("name", "Anonymous"),
            "email": fb.get("email"),
            "message": fb.get("message"),
            "date": fb.get("date").strftime("%Y-%m-%d %H:%M:%S") if fb.get("date") else "Unknown"
        })
    return feedbacks

@router.get("/users")
def get_all_users(user: dict = Depends(admin_only)):
    users = list(users_col.find({}, {"password": 0})) # Don't send passwords
    return [sanitize_item(u) for u in users]

@router.post("/toggle_premium/{user_id}")
def toggle_premium(user_id: str, admin: dict = Depends(admin_only)):
    try:
        user = users_col.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        new_status = not user.get("is_premium", False)
        users_col.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"is_premium": new_status}}
        )
        return {"msg": f"Premium status updated to {new_status}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/approve/{item_id}")
def approve_item(item_id: str, user: dict = Depends(admin_only)):
    try:
        result = items_col.update_one(
            {"_id": ObjectId(item_id)},
            {"$set": {"is_claimed": True}}
        )
        if result.modified_count == 1:
            return {"msg": "Item marked as claimed"}
        raise HTTPException(status_code=404, detail="Item not found")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid item ID")

@router.delete("/delete/{item_id}")
def delete_item(item_id: str, user: dict = Depends(admin_only)):
    try:
        result = items_col.delete_one({"_id": ObjectId(item_id)})
        if result.deleted_count == 1:
            return {"msg": "Item deleted successfully"}
        raise HTTPException(status_code=404, detail="Item not found")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid item ID")

@router.post("/update/{item_id}")
def update_item(item_id: str, updates: dict, user: dict = Depends(admin_only)):
    try:
        # Prevent manual ID update
        if "_id" in updates: del updates["_id"]
        result = items_col.update_one(
            {"_id": ObjectId(item_id)},
            {"$set": updates}
        )
        if result.modified_count == 1:
            return {"msg": "Item updated successfully"}
        raise HTTPException(status_code=404, detail="Item not found or no changes made")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid item ID or data")
