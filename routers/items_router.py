from fastapi import APIRouter, Depends, HTTPException, Form, UploadFile, File, Query
from typing import Optional
import os
import uuid
from io import BytesIO
from PIL import Image
from bson import ObjectId
from bson.errors import InvalidId
from datetime import datetime
from database import items_col
from ai_matcher import (
    match_with_gemini, match_with_tfidf, match_with_embeddings,
    generate_qr_for_item, generate_image_description, generate_local_description,
    get_image_embedding, run_ensemble_matching, query_rag_agent, geocode_location
)
from routers.auth import get_current_user
from models import Item

def compress_and_save_image(upload_file, target_path, max_size=(800, 800), quality=85):
    try:
        content = upload_file.file.read()
        upload_file.file.seek(0)
        
        img = Image.open(BytesIO(content))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        img.save(target_path, "JPEG", quality=quality, optimize=True)
        return True
    except Exception as e:
        print(f"Error compressing image: {e}")
        return False

router = APIRouter(tags=["items"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/report_found")
async def report_found(
    item_name: str = Form(...),
    description: str = Form(...),
    date: str = Form(...),
    time: str = Form(...),
    location: str = Form(...),
    contact_info: str = Form(...),
    priority: bool = Form(False),
    image: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    user: dict = Depends(get_current_user)
):
    filename = f"{uuid.uuid4()}_{image.filename.replace(' ', '_')}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Compress and save image, fall back to raw bytes if failure
    if not compress_and_save_image(image, file_path):
        image.file.seek(0)
        with open(file_path, "wb") as f:
            content = await image.read()
            f.write(content)

    image_url = f"/uploads/{filename}"

    # Generate local embedding for similarity matching
    embedding = get_image_embedding(file_path)

    # Resolve coordinates via geocode if missing
    lat_val = latitude
    lon_val = longitude
    if lat_val is None or lon_val is None:
        coords = geocode_location(location)
        if coords:
            lat_val, lon_val = coords

    item = {
        "item_name": item_name,
        "description": description,
        "date": date,
        "time": time,
        "location": location,
        "latitude": lat_val,
        "longitude": lon_val,
        "contact_info": contact_info,
        "priority": priority,
        "image_url": image_url,
        "type": "found",
        "is_claimed": False,
        "email": user["email"],
        "embedding": embedding.tolist() if embedding is not None else None
    }
    result = items_col.insert_one(item)
    item_id = str(result.inserted_id)

    # Insert into local LSH Vector Database for sub-linear search
    from vector_db import vector_db
    vector_db.insert_item(
        item_id=item_id,
        image_vector=item["embedding"],
        text_string=f"{item_name} {description} {location}".lower()
    )

    # Run Matching Logic
    try:
        if priority:
            print("Premium Mode (Found): Matching with Gemini...")
            match_with_gemini(item)
        else:
            print("Standard Mode (Found): Matching with Multi-Modal Ensemble...")
            run_ensemble_matching(item)
    except Exception as e:
        print("Matching failed for found item:", e)

    return {
        "message": "Found item reported successfully",
        "item_id": item_id,
        "generate_qr": True
    }

@router.post("/report_lost")
async def report_lost(
    item_name: str = Form(...),
    description: str = Form(...),
    date: str = Form(...),
    time: str = Form(...),
    location: str = Form(...),
    contact_info: str = Form(...),
    priority: bool = Form(False),
    image: UploadFile = File(...),
    wants_call: bool = Form(False),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    user: dict = Depends(get_current_user)
):
    filename = f"{uuid.uuid4()}_{image.filename.replace(' ', '_')}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Compress and save image, fall back to raw bytes if failure
    if not compress_and_save_image(image, file_path):
        image.file.seek(0)
        with open(file_path, "wb") as f:
            content = await image.read()
            f.write(content)

    image_url = f"/uploads/{filename}"

    # Generate local embedding for similarity matching
    embedding = get_image_embedding(file_path)

    # Resolve coordinates via geocode if missing
    lat_val = latitude
    lon_val = longitude
    if lat_val is None or lon_val is None:
        coords = geocode_location(location)
        if coords:
            lat_val, lon_val = coords

    item = {
        "item_name": item_name,
        "description": description,
        "date": date,
        "time": time,
        "location": location,
        "latitude": lat_val,
        "longitude": lon_val,
        "contact_info": contact_info,
        "priority": priority,
        "wants_call": wants_call,
        "image_url": image_url,
        "type": "lost",
        "is_claimed": False,
        "email": user["email"],
        "embedding": embedding.tolist() if embedding is not None else None
    }

    result = items_col.insert_one(item)
    item_id = str(result.inserted_id)

    # Insert into local LSH Vector Database for sub-linear search
    from vector_db import vector_db
    vector_db.insert_item(
        item_id=item_id,
        image_vector=item["embedding"],
        text_string=f"{item_name} {description} {location}".lower()
    )

    # Run Matching Logic
    try:
        if priority:
            print("Premium Mode: Matching with Gemini...")
            match_with_gemini(item)
        else:
            print("Standard Mode: Matching with Multi-Modal Ensemble...")
            run_ensemble_matching(item)
    except Exception as e:
        print("Matching failed:", e)

    return {
        "message": "Lost item reported successfully.",
        "item_id": item_id,
        "wants_call": wants_call,
        "generate_qr": True
    }

@router.get("/stats")
def get_stats():
    total = items_col.count_documents({})
    found = items_col.count_documents({"type": "found"})
    lost = items_col.count_documents({"type": "lost"})
    return {
        "total_items": total,
        "found_items": found,
        "lost_items": lost
    }

@router.get("/browse")
def get_unclaimed_found_items(user_email: Optional[str] = Query(None)):
    query = {"type": "found", "is_claimed": False}
    if user_email:
        query["email"] = {"$ne": user_email}

    items = list(items_col.find(query))
    for item in items:
        item["_id"] = str(item["_id"])
    return items

@router.get("/user/dashboard")
def get_dashboard(user: dict = Depends(get_current_user)):
    email = user["email"]
    lost_reports = list(items_col.find({"email": email, "type": "lost"}))
    found_reports = list(items_col.find({"email": email, "type": "found"}))

    for r in lost_reports + found_reports:
        r["_id"] = str(r["_id"])

    return {
        "lost_reports": lost_reports,
        "found_reports": found_reports
    }

@router.delete("/items/{item_id}")
def delete_item(item_id: str, user: dict = Depends(get_current_user)):
    try:
        result = items_col.delete_one({
            "_id": ObjectId(item_id),
            "email": user["email"]
        })
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Item not found or not authorized")
        return {"message": "Item deleted"}
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid item ID")

@router.get("/can_submit")
def can_submit(user: dict = Depends(get_current_user)):
    from database import users_col
    email = user["email"]
    
    # Check if user is admin or has a limit bypass
    db_user = users_col.find_one({"email": email})
    if db_user and (db_user.get("is_admin") or db_user.get("bypass_limit")):
        print(f"Bypass: Limit suppressed for test/admin account: {email}")
        return {"can_submit": True}

    today = datetime.now().strftime("%Y-%m-%d")
    count = items_col.count_documents({
        "email": email,
        "date": today
    })
    
    if count >= 100:  # Increased from 3 to 100 for testing
        return {"can_submit": False, "message": "You’ve reached today’s limit (100 reports). Try again tomorrow."}
    return {"can_submit": True}

@router.get("/items/{item_id}")
def get_item_details(item_id: str):
    try:
        item = items_col.find_one({"_id": ObjectId(item_id)})
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        
        # Convert ObjectId to string and remove sensitive info if any
        item["_id"] = str(item["_id"])
        # We might want to mask contact info if not claimed, but for QR scan it's usually public info for lost items
        return item
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid item ID")

@router.get("/generate_qr/{item_id}")
def get_qr_api(item_id: str):
    qr = generate_qr_for_item(item_id)
    return {"qr": qr}

@router.post("/agent_request")
def agent_assist(item_id: str):
    import google.generativeai as genai
    model = genai.GenerativeModel(model_name="gemini-flash-latest")
    try:
        item = items_col.find_one({"_id": ObjectId(item_id)})
        if not item:
            raise HTTPException(404, detail="Item not found")
    except InvalidId:
        raise HTTPException(400, detail="Invalid item ID format")

    # Fixed: items_col response needs description
    desc = item.get('description', 'No description provided')

    response = model.generate_content(f"""
    You are an expert AI lost-and-found assistant. The user reported:
    {desc}
    
    Based on this, suggest any additional details they could provide to improve matching.
    """)
    return {"agent_response": response.text.strip()}

@router.post("/claim/{item_id}")
async def claim_item(
    item_id: str,
    name: str = Form(...),
    contact: str = Form(...),
    proof: str = Form("")
):
    from notif import send_email
    try:
        item = items_col.find_one({"_id": ObjectId(item_id)})
        if not item:
            raise HTTPException(status_code=404, detail="Item not found.")

        found_user_email = item.get("email") or item.get("contact_info", "admin@lostlink.ai")
        subject = f"[LostLink] Claim Request for: {item.get('item_name','Item')}"
        message = f"""
        Someone has claimed the item you reported as FOUND!

        Claimer Name: {name}
        Contact: {contact}
        Proof: {proof or "Not provided"}

        Please reach out to verify.
        """

        send_email(to=found_user_email, subject=subject, body=message)
        return {"message": "Claim sent to item reporter."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/describe_image")
async def describe_image(
    image: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
    """
    Auto-switches between Gemini (Premium) and MobileNet (Standard).
    """
    from database import users_col
    db_user = users_col.find_one({"email": user["email"]})
    is_premium = db_user.get("is_premium", False) if db_user else False

    temp_filename = f"temp_{uuid.uuid4()}_{image.filename}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)
    
    try:
        with open(temp_path, "wb") as f:
            f.write(await image.read())
        
        if is_premium:
            print(f"Premium User ({user['email']}): Using Gemini...")
            description = generate_image_description(temp_path)
            mode = "premium"
        else:
            print(f"Standard User ({user['email']}): Using MobileNet...")
            description = generate_local_description(temp_path)
            mode = "standard"
            
        if not description:
            raise HTTPException(status_code=500, detail="Failed to generate description")
            
        return {"description": description, "mode": mode}
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@router.post("/chat")
async def chat_rag(message: str = Form(...)):
    if not message.strip():
        raise HTTPException(status_code=400, detail="Empty query message.")
    response_text = query_rag_agent(message)
    return {"response": response_text}
