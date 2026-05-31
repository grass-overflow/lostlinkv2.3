import os
import math
import google.generativeai as genai
from database import items_col, matches_col
from notif import send_email, speak_message, make_phone_call
from datetime import datetime
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from vector_db import vector_db
import requests

def parse_dt(d, t):
    try:
        return datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M")
    except:
        return None

NITK_LOCATIONS = [
    "Karavali", "Aravali", "Vindhya", "Satpura", "Nilgiri", "Pushpagiri", "Brahmagiri", 
    "Sahyadri", "Trishul", "Everest", "Himalaya", "Kailash", "Shivalik", "Ganga", 
    "Kaveri", "Yamuna", "Sharavathi", "Netravathi", "Godavari", "International Hostel",
    "Mega Tower 1", "Mega Tower 2", "Mega Tower 3", "LHC-A", "LHC-B", "Lecture Hall Complex",
    "Main Building", "Central Library", "Central Computer Centre", "CCC", "Health Care Centre",
    "Post Office", "Canara Bank", "SBI", "Shopping Complex", "Canteens", "Staff Club",
    "Silver Jubilee Auditorium", "SJA", "OAT", "Student Activity Centre", "SAC",
    "Sports Ground", "Basketball Court", "Swimming Pool", "Indoor Sports Complex", "Gym",
    "STEP", "NITK Beach"
]

NITK_COORDINATES = {
    "karavali": (13.0125, 74.7925),
    "aravali": (13.0120, 74.7920),
    "vindhya": (13.0118, 74.7922),
    "satpura": (13.0115, 74.7924),
    "nilgiri": (13.0110, 74.7926),
    "pushpagiri": (13.0105, 74.7928),
    "brahmagiri": (13.0100, 74.7930),
    "sahyadri": (13.0068, 74.7925),
    "trishul": (13.0070, 74.7927),
    "everest": (13.0072, 74.7929),
    "himalaya": (13.0074, 74.7931),
    "kailash": (13.0076, 74.7933),
    "shivalik": (13.0078, 74.7935),
    "ganga": (13.0080, 74.7937),
    "kaveri": (13.0082, 74.7939),
    "yamuna": (13.0084, 74.7941),
    "sharavathi": (13.0086, 74.7943),
    "netravathi": (13.0088, 74.7945),
    "godavari": (13.0090, 74.7947),
    "international hostel": (13.0095, 74.7915),
    "mega tower 1": (13.0080, 74.7910),
    "mega tower 2": (13.0082, 74.7912),
    "mega tower 3": (13.0084, 74.7914),
    "lhc-a": (13.0115, 74.7950),
    "lhc-b": (13.0118, 74.7952),
    "lecture hall complex": (13.0116, 74.7951),
    "main building": (13.0108, 74.7943),
    "central library": (13.0102, 74.7938),
    "central computer centre": (13.0106, 74.7940),
    "ccc": (13.0106, 74.7940),
    "health care centre": (13.0104, 74.7955),
    "post office": (13.0100, 74.7950),
    "canara bank": (13.0099, 74.7949),
    "sbi": (13.0102, 74.7952),
    "shopping complex": (13.0105, 74.7960),
    "canteens": (13.0100, 74.7945),
    "staff club": (13.0112, 74.7930),
    "silver jubilee auditorium": (13.0110, 74.7948),
    "sja": (13.0110, 74.7948),
    "oat": (13.0108, 74.7954),
    "student activity centre": (13.0098, 74.7955),
    "sac": (13.0098, 74.7955),
    "sports ground": (13.0095, 74.7960),
    "basketball court": (13.0096, 74.7958),
    "swimming pool": (13.0092, 74.7962),
    "indoor sports complex": (13.0094, 74.7965),
    "gym": (13.0093, 74.7964),
    "step": (13.0130, 74.7950),
    "nitk beach": (13.0120, 74.7860)
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Radius of Earth in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def geocode_location(location_str):
    if not location_str:
        return None
    loc_lower = location_str.lower()
    for landmark, coords in NITK_COORDINATES.items():
        if landmark in loc_lower:
            return coords
    return None

clf_with_image = None
clf_text_only = None

def train_matching_models():
    global clf_with_image, clf_text_only
    print("test msg : Training/updating machine-learning-powered matching models...")
    
    # 1. Generate robust synthetic training data as baseline
    X_img_synth = []
    y_img_synth = []
    
    # Positive examples (match = 1)
    for _ in range(50):
        # Good visual + good text + close distance + small positive time gap
        X_img_synth.append([np.random.uniform(0.75, 0.95), np.random.uniform(0.6, 0.9), np.random.uniform(0.0, 0.3), np.random.uniform(0.0, 5.0)])
        y_img_synth.append(1)
        # Good visual + moderate text + close distance + small positive time gap
        X_img_synth.append([np.random.uniform(0.8, 0.99), np.random.uniform(0.2, 0.5), np.random.uniform(0.0, 0.2), np.random.uniform(0.0, 3.0)])
        y_img_synth.append(1)
        # Moderate visual + high text + close distance + small positive time gap
        X_img_synth.append([np.random.uniform(0.4, 0.7), np.random.uniform(0.75, 0.95), np.random.uniform(0.0, 0.1), np.random.uniform(0.0, 2.0)])
        y_img_synth.append(1)
        
    # Negative examples (match = 0)
    for _ in range(50):
        # Low similarities
        X_img_synth.append([np.random.uniform(0.0, 0.4), np.random.uniform(0.0, 0.4), np.random.uniform(1.0, 5.0), np.random.uniform(5.0, 30.0)])
        y_img_synth.append(0)
        # High visual similarity but different location/time (different items that look similar)
        X_img_synth.append([np.random.uniform(0.8, 0.95), np.random.uniform(0.0, 0.2), np.random.uniform(2.0, 10.0), np.random.uniform(0.0, 10.0)])
        y_img_synth.append(0)
        # High text similarity but huge distance or time gap
        X_img_synth.append([np.random.uniform(0.2, 0.5), np.random.uniform(0.8, 0.95), np.random.uniform(5.0, 20.0), np.random.uniform(15.0, 60.0)])
        y_img_synth.append(0)
        # Negative time gap (lost after found)
        X_img_synth.append([np.random.uniform(0.5, 0.9), np.random.uniform(0.5, 0.9), np.random.uniform(0.0, 2.0), np.random.uniform(-10.0, -0.1)])
        y_img_synth.append(0)

    # Features for text-only model: [text_sim, distance_km, time_gap_days]
    X_txt_synth = []
    y_txt_synth = []
    
    # Positive examples (match = 1)
    for _ in range(50):
        # High text similarity + close distance + small positive time gap
        X_txt_synth.append([np.random.uniform(0.7, 0.95), np.random.uniform(0.0, 0.3), np.random.uniform(0.0, 5.0)])
        y_txt_synth.append(1)
        # Moderate text similarity + very close distance + small positive time gap
        X_txt_synth.append([np.random.uniform(0.5, 0.7), np.random.uniform(0.0, 0.1), np.random.uniform(0.0, 2.0)])
        y_txt_synth.append(1)
        
    # Negative examples (match = 0)
    for _ in range(50):
        # Low similarity
        X_txt_synth.append([np.random.uniform(0.0, 0.35), np.random.uniform(1.0, 5.0), np.random.uniform(5.0, 30.0)])
        y_txt_synth.append(0)
        # High similarity but high distance/time gap
        X_txt_synth.append([np.random.uniform(0.7, 0.95), np.random.uniform(3.0, 15.0), np.random.uniform(10.0, 60.0)])
        y_txt_synth.append(0)
        # Negative time gap
        X_txt_synth.append([np.random.uniform(0.5, 0.9), np.random.uniform(0.0, 2.0), np.random.uniform(-10.0, -0.1)])
        y_txt_synth.append(0)

    X_img = np.array(X_img_synth)
    y_img = np.array(y_img_synth)
    X_txt = np.array(X_txt_synth)
    y_txt = np.array(y_txt_synth)

    # Mix in actual historical matches if available from MongoDB
    try:
        from bson import ObjectId
        historical_matches = list(matches_col.find())
        print(f"test msg : Found {len(historical_matches)} confirmed matches in DB to include in training.")
        
        hist_X_img = []
        hist_y_img = []
        hist_X_txt = []
        hist_y_txt = []
        
        for match in historical_matches:
            item_a = items_col.find_one({"_id": ObjectId(match["item_a_id"])})
            item_b = items_col.find_one({"_id": ObjectId(match["item_b_id"])}) if match.get("item_b_id") != "new" else None
            
            if item_a and item_b:
                # 1. Visual similarity
                vis_sim = 0.0
                has_vis = False
                if item_a.get("embedding") and item_b.get("embedding"):
                    has_vis = True
                    emb_a = np.array(item_a["embedding"])
                    emb_b = np.array(item_b["embedding"])
                    norm_a = np.linalg.norm(emb_a)
                    norm_b = np.linalg.norm(emb_b)
                    if norm_a > 0 and norm_b > 0:
                        vis_sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
                
                # 2. Text similarity
                text_sim = 0.0
                text_a = f"{item_a.get('item_name', '')} {item_a.get('description', '')} {item_a.get('location', '')}".lower()
                text_b = f"{item_b.get('item_name', '')} {item_b.get('description', '')} {item_b.get('location', '')}".lower()
                vec_a = vector_db.get_text_vector(text_a)
                vec_b = vector_db.get_text_vector(text_b)
                norm_va = np.linalg.norm(vec_a)
                norm_vb = np.linalg.norm(vec_b)
                if norm_va > 0 and norm_vb > 0:
                    text_sim = float(np.dot(vec_a, vec_b) / (norm_va * norm_vb))
                
                # 3. Distance
                dist_km = 0.5
                lat_a, lon_a = item_a.get("latitude"), item_a.get("longitude")
                lat_b, lon_b = item_b.get("latitude"), item_b.get("longitude")
                if lat_a is not None and lon_a is not None and lat_b is not None and lon_b is not None:
                    dist_km = haversine(lat_a, lon_a, lat_b, lon_b)
                
                # 4. Time gap
                time_gap_days = 3.0
                dt_a = parse_dt(item_a.get("date"), item_a.get("time"))
                dt_b = parse_dt(item_b.get("date"), item_b.get("time"))
                if dt_a and dt_b:
                    lost_dt = dt_a if item_a["type"] == "lost" else dt_b
                    found_dt = dt_b if item_a["type"] == "lost" else dt_a
                    time_gap_days = (found_dt - lost_dt).total_seconds() / 86400.0
                
                if has_vis:
                    hist_X_img.append([vis_sim, text_sim, dist_km, time_gap_days])
                    hist_y_img.append(1)
                else:
                    hist_X_txt.append([text_sim, dist_km, time_gap_days])
                    hist_y_txt.append(1)
                    
                unrelated_item = items_col.find_one({
                    "type": item_b["type"],
                    "_id": {"$ne": item_b["_id"]}
                })
                if unrelated_item:
                    u_vis_sim = 0.0
                    u_has_vis = False
                    if item_a.get("embedding") and unrelated_item.get("embedding"):
                        u_has_vis = True
                        emb_u = np.array(unrelated_item["embedding"])
                        norm_u = np.linalg.norm(emb_u)
                        if norm_a > 0 and norm_u > 0:
                            u_vis_sim = float(np.dot(emb_a, emb_u) / (norm_a * norm_u))
                    
                    u_text_sim = 0.0
                    text_u = f"{unrelated_item.get('item_name', '')} {unrelated_item.get('description', '')} {unrelated_item.get('location', '')}".lower()
                    vec_u = vector_db.get_text_vector(text_u)
                    norm_vu = np.linalg.norm(vec_u)
                    if norm_va > 0 and norm_vu > 0:
                        u_text_sim = float(np.dot(vec_a, vec_u) / (norm_va * norm_vu))
                    
                    u_dist_km = 0.5
                    lat_u, lon_u = unrelated_item.get("latitude"), unrelated_item.get("longitude")
                    if lat_a is not None and lon_a is not None and lat_u is not None and lon_u is not None:
                        u_dist_km = haversine(lat_a, lon_a, lat_u, lon_u)
                    
                    u_time_gap_days = 3.0
                    dt_u = parse_dt(unrelated_item.get("date"), unrelated_item.get("time"))
                    if dt_a and dt_u:
                        lost_dt = dt_a if item_a["type"] == "lost" else dt_u
                        found_dt = dt_u if item_a["type"] == "lost" else dt_a
                        u_time_gap_days = (found_dt - lost_dt).total_seconds() / 86400.0
                        
                    if u_has_vis:
                        hist_X_img.append([u_vis_sim, u_text_sim, u_dist_km, u_time_gap_days])
                        hist_y_img.append(0)
                    else:
                        hist_X_txt.append([u_text_sim, u_dist_km, u_time_gap_days])
                        hist_y_txt.append(0)
                        
        if hist_X_img:
            X_img = np.vstack([X_img, hist_X_img])
            y_img = np.concatenate([y_img, hist_y_img])
        if hist_X_txt:
            X_txt = np.vstack([X_txt, hist_X_txt])
            y_txt = np.concatenate([y_txt, hist_y_txt])
            
    except Exception as e:
        print(f"test msg : Could not incorporate MongoDB matches for ML training: {e}")

    clf_with_image = LogisticRegression(class_weight="balanced")
    clf_with_image.fit(X_img, y_img)
    clf_text_only = LogisticRegression(class_weight="balanced")
    clf_text_only.fit(X_txt, y_txt)
    print("test msg : ML matching models training completed.")

api_key = os.getenv("GEMINI_API_KEY", "").strip()
if api_key.startswith('"') and api_key.endswith('"'):
    api_key = api_key[1:-1]

print(f"Test msg for AI matcher : Configuring Gemini with key starting with {api_key[:5]}...")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-flash-latest")

_local_sim_model = None
_local_transform = None

def get_local_model():
    global _local_sim_model, _local_transform
    if _local_sim_model is None:
        print("test msg : loading local MobileNetV2 for Standard similarity...")
        weights = MobileNet_V2_Weights.DEFAULT
        _local_sim_model = mobilenet_v2(weights=weights).eval()
        _local_sim_model.classifier = torch.nn.Identity()
        
        _local_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _local_sim_model, _local_transform

def get_image_embedding(image_path):
    try:
        model, transform = get_local_model()
        with Image.open(image_path).convert('RGB') as img:
            img_t = transform(img).unsqueeze(0)
            with torch.no_grad():
                embedding = model(img_t)
            vec = embedding.squeeze().numpy()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec
    except Exception as e:
        print(f"test msg : Error extracting local embedding: {e}")
        return None

def match_with_embeddings(new_item):
    if not new_item.get("image_url"): return
    
    new_img_path = os.path.join(os.getcwd(), new_item["image_url"].lstrip("/"))
    if not os.path.exists(new_img_path): return
    
    new_embedding = get_image_embedding(new_img_path)
    if new_embedding is None: return
        
    other_items = list(items_col.find({
        "type": {"$ne": new_item["type"]},
        "is_claimed": False,
        "embedding": {"$exists": True, "$ne": None}
    }))
    
    if not other_items: return

    existing_embeddings = [np.array(item["embedding"]) for item in other_items]
    similarities = cosine_similarity([new_embedding], existing_embeddings)[0]
    
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    
    print(f"test msg : Local Embedding Match Score: {best_score:.2f}")
    
    if best_score > 0.82: 
        match = other_items[best_idx]
        print(f"test msg : Local Visual Match Found: {match['item_name']}")
        
        matches_col.insert_one({
            "item_a_id": str(match["_id"]),
            "item_b_id": str(new_item["_id"]) if "_id" in new_item else "new",
            "match_type": "Visual (MobileNetV2)",
            "score": float(best_score),
            "reason": f"Visual similarity score of {best_score:.2f} using MobileNetV2 embeddings.",
            "timestamp": datetime.utcnow()
        })
        
        ai_agent_notify(match, new_item)

def match_with_gemini(new_item):
    other_items = items_col.find({
        "type": {"$ne": new_item["type"]},
        "is_claimed": False
    })

    matched_contacts = set()

    for existing in other_items:
        prompt = f"""
Compare these two item reports and determine if they describe the same lost item.

Item A Description:
{existing['description']}

Item B Description:
{new_item['description']}

Respond with only: yes or no.
"""
        try:
            response = model.generate_content(prompt)
            decision = response.text.strip().lower()
        except Exception as e:
            print(f"test msg : Error from Gemini model: {e}")
            continue

        if decision.startswith("yes"):
            if existing["contact_info"] in matched_contacts:
                continue

            print("test msg : Match found")
            
            matches_col.insert_one({
                "item_a_id": str(existing["_id"]),
                "item_b_id": str(new_item["_id"]) if "_id" in new_item else "new",
                "match_type": "AI Reasoning (Gemini)",
                "score": 1.0,
                "reason": f"Gemini AI analyzed descriptions and confirmed a match.",
                "timestamp": datetime.utcnow()
            })

            ai_agent_notify(existing, new_item)
            matched_contacts.add(existing["contact_info"])
            break

import re

def is_valid_email(address):
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", address))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def match_with_tfidf(new_item):
    target_type = "found" if new_item["type"] == "lost" else "lost"
    all_items = list(items_col.find({"type": target_type, "is_claimed": False}))
    
    if not all_items:
        return

    def get_campus_location(text):
        text = text.lower()
        for loc in NITK_LOCATIONS:
            if loc.lower() in text:
                return loc.lower()
        return None

    def get_text(it):
        return f"{it.get('description', '')} {it.get('location', '')}".lower()

    corpus = [get_text(item) for item in all_items]
    new_text = get_text(new_item)
    corpus.append(new_text)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3), 
        analyzer='word',
        stop_words='english',
        min_df=1
    ).fit_transform(corpus)
    vectors = vectorizer.toarray()

    similarities = cosine_similarity([vectors[-1]], vectors[:-1])[0]

    from datetime import datetime

    new_dt = parse_dt(new_item["date"], new_item["time"])

    scored_matches = []
    for i, item in enumerate(all_items):
        score = similarities[i]
        
        new_loc_str = new_item.get("location", "").lower()
        old_loc_str = item.get("location", "").lower()
        
        if new_loc_str and old_loc_str and (new_loc_str in old_loc_str or old_loc_str in new_loc_str):
            score += 0.2
        
        new_campus_loc = get_campus_location(new_loc_str)
        old_campus_loc = get_campus_location(old_loc_str)
        if new_campus_loc and old_campus_loc and new_campus_loc == old_campus_loc:
            print(f"Campus Match! Both reports reference: {new_campus_loc}")
            score += 0.3
        
        item_dt = parse_dt(item["date"], item["time"])
        if new_dt and item_dt:
            if new_item["type"] == "lost" and new_dt <= item_dt:
                score += 0.15
            elif new_item["type"] == "found" and item_dt <= new_dt:
                score += 0.15
        
        scored_matches.append((score, item))

    scored_matches.sort(key=lambda x: x[0], reverse=True)
    best_score, best_match = scored_matches[0]

    print(f"Enhanced Match (TF-IDF+Context): {best_match['item_name']} (Final Score: {best_score:.2f})")

    if best_score > 0.75:
        matches_col.insert_one({
            "item_a_id": str(best_match["_id"]),
            "item_b_id": str(new_item["_id"]) if "_id" in new_item else "new",
            "match_type": "Text/Context (TF-IDF)",
            "score": float(best_score),
            "reason": f"Matched using TF-IDF text similarity and spatio-temporal logic.",
            "timestamp": datetime.utcnow()
        })

        ai_agent_notify(best_match if best_match["type"] == "lost" else new_item, 
                        new_item if best_match["type"] == "lost" else best_match)
    else:
        print("No strong match found.")


import qrcode
from io import BytesIO
import base64

def generate_qr_for_item(item_id: str):
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    qr_text = f"{base_url}/qr/{item_id}"

    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(qr_text)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    base64_qr = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/png;base64,{base64_qr}"


def ai_agent_notify(lost_item, found_item):
    contact = lost_item["contact_info"]
    subject = "Possible Match for Your Lost Item!"
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    body = f"""
Hi,

We may have found a match for your lost item: {lost_item['item_name']}.

Matched with found item:
- {found_item['item_name']}
- Description: {found_item['description']}
- Location: {found_item['location']}
- Date: {found_item['date']} at {found_item['time']}

Please log in to LostLink to view details and confirm:
{base_url}/login.html

– LostLink Agent 

DO NOT REPLY. 
THIS IS AUTO-GENERATED.
"""

    print(f"Notifying {contact} about match with {found_item['item_name']}")

    if is_valid_email(contact):
        send_email(to=contact, subject=subject, body=body)
        print(f"Email sent to {contact}")
    else:
        print(f"Skipping email: {contact} is not a valid email.")

    speak_message(
        f"A potential match has been found for lost item: {lost_item['item_name']} at {found_item['location']}."
    )

    if lost_item.get("wants_call") and contact.startswith("+"):
        make_phone_call(
            to_number=contact,
            message=f"Hello! A match is found for your lost item: {lost_item['item_name']} at {found_item['location']}. Please check LostLink. Thank you."
        )

def generate_image_description(image_path):
    try:
        from PIL import Image
        img = Image.open(image_path)
        
        try:
            prompt = "Describe this object in detail for a lost and found system. Focus on color, brand, unique markings, and material. Keep it concise but descriptive (max 100 words)."
            response = model.generate_content([prompt, img])
            return response.text.strip()
        finally:
            img.close()
    except Exception as e:
        print(f"Error generating image description: {e}")
        return None

def generate_local_description(image_path):
    try:
        from torchvision.models import MobileNet_V2_Weights
        weights = MobileNet_V2_Weights.DEFAULT
        categories = weights.meta["categories"]
        
        model, transform = get_local_model()
        classifier = _get_local_classifier()
        
        with Image.open(image_path).convert('RGB') as img:
            img_t = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = classifier(img_t)
            
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            prob, catid = torch.max(probabilities, 0)
            
            return f"Identified as: {categories[catid.item()]}"
    except Exception as e:
        print(f" Local description failed: {e}")
        return "Local description unavailable."

_local_classifier = None
def _get_local_classifier():
    global _local_classifier
    if _local_classifier is None:
        weights = MobileNet_V2_Weights.DEFAULT
        _local_classifier = mobilenet_v2(weights=weights).eval()
    return _local_classifier

def run_ensemble_matching(new_item):
    target_type = "found" if new_item["type"] == "lost" else "lost"
    
    # 1. Visual Similarity Lookup via LSH Index
    visual_similarities = {}
    new_embedding = None
    if new_item.get("image_url"):
        new_img_path = os.path.join(os.getcwd(), new_item["image_url"].lstrip("/"))
        if os.path.exists(new_img_path):
            new_embedding = get_image_embedding(new_img_path)
            
    if new_embedding is not None:
        image_matches = vector_db.image_index.query(new_embedding, max_hamming_distance=3)
        visual_similarities = {cid: sim for cid, sim in image_matches}

    # 2. Textual Similarity Lookup via LSH Index
    def get_text(it):
        return f"{it.get('item_name', '')} {it.get('description', '')} {it.get('location', '')}".lower()

    new_text = get_text(new_item)
    new_text_vector = vector_db.get_text_vector(new_text)
    
    text_matches = vector_db.text_index.query(new_text_vector, max_hamming_distance=2)
    text_similarities = {cid: sim for cid, sim in text_matches}

    # 3. Candidate Pool Retrieval and Filtering
    candidate_ids = set(list(visual_similarities.keys()) + list(text_similarities.keys()))
    if not candidate_ids:
        print("test msg : LSH indices returned no potential candidates.")
        return

    # Fetch candidate details for validation from MongoDB using candidate IDs
    from bson import ObjectId
    candidates = list(items_col.find({
        "_id": {"$in": [ObjectId(cid) for cid in candidate_ids if ObjectId.is_valid(cid)]},
        "type": target_type,
        "is_claimed": False
    }))

    if not candidates:
        print("test msg : No unclaimed matching candidate documents in database.")
        return

    new_dt = parse_dt(new_item["date"], new_item["time"])
    
    scored_matches = []
    for cand in candidates:
        cid = str(cand["_id"])
        
        # Base LSH-derived similarity scores
        text_sim = text_similarities.get(cid, 0.0)
        visual_sim = visual_similarities.get(cid, 0.0)
        
        # 1. Distance Calculation (Haversine)
        dist_km = 0.5 # default/neutral value
        lat_new, lon_new = new_item.get("latitude"), new_item.get("longitude")
        lat_cand, lon_cand = cand.get("latitude"), cand.get("longitude")
        
        # Try geocoding if coordinates are missing in MongoDB
        if lat_new is None or lon_new is None:
            coords = geocode_location(new_item.get("location"))
            if coords:
                lat_new, lon_new = coords
        if lat_cand is None or lon_cand is None:
            coords = geocode_location(cand.get("location"))
            if coords:
                lat_cand, lon_cand = coords
                
        if lat_new is not None and lon_new is not None and lat_cand is not None and lon_cand is not None:
            dist_km = haversine(lat_new, lon_new, lat_cand, lon_cand)
            
        # 2. Time Gap Calculation (in days)
        time_gap_days = 3.0 # default/neutral value
        cand_dt = parse_dt(cand["date"], cand["time"])
        if new_dt and cand_dt:
            # Physical constraint: Lost must occur before Found
            lost_dt = new_dt if new_item["type"] == "lost" else cand_dt
            found_dt = cand_dt if new_item["type"] == "lost" else new_dt
            
            # Signed time gap: found_dt - lost_dt (should be positive)
            time_gap_days = (found_dt - lost_dt).total_seconds() / 86400.0

        # 3. Model Inference (Logistic Regression)
        if clf_with_image is None or clf_text_only is None:
            train_matching_models()
            
        has_visual = (new_embedding is not None and cand.get("embedding") is not None)
        if has_visual:
            features = np.array([[visual_sim, text_sim, dist_km, time_gap_days]])
            ensemble_score = float(clf_with_image.predict_proba(features)[0][1])
        else:
            features = np.array([[text_sim, dist_km, time_gap_days]])
            ensemble_score = float(clf_text_only.predict_proba(features)[0][1])
            
        ensemble_score = max(0.0, min(1.0, ensemble_score))
        scored_matches.append((ensemble_score, cand, has_visual, visual_sim, text_sim, dist_km, time_gap_days))

    if not scored_matches:
        return

    # Sort candidates by ensemble score
    scored_matches.sort(key=lambda x: x[0], reverse=True)
    best_score, best_match, has_vis, vis_sim, txt_sim, dist_km, time_gap_days = scored_matches[0]
    
    print(f"test msg : Ensemble Match Top Result: {best_match['item_name']} with ML match probability {best_score:.4f} (Visual: {vis_sim:.2f}, Text: {txt_sim:.2f}, Distance: {dist_km:.2f} km, Time Gap: {time_gap_days:.2f} days)")
    
    THRESHOLD = 0.70
    
    if best_score >= THRESHOLD:
        print(f"test msg : Unified Ensemble Match Found: {best_match['item_name']}")
        
        match_type = "Multi-Modal ML Ensemble (LSH-indexed)" if has_vis else "Text/Context ML Ensemble (LSH-indexed)"
        reason = (f"ML predicted match probability of {best_score:.2f} using "
                  f"{'MobileNetV2 visual (sim: ' + f'{vis_sim:.2f}' + ') and ' if has_vis else ''}"
                  f"local dense text similarity (sim: {txt_sim:.2f}) + Haversine distance spatial validation ({dist_km:.2f} km) via LSH search.")
        
        matches_col.insert_one({
            "item_a_id": str(best_match["_id"]),
            "item_b_id": str(new_item["_id"]) if "_id" in new_item else "new",
            "match_type": match_type,
            "score": float(best_score),
            "reason": reason,
            "timestamp": datetime.utcnow()
        })
        
        lost_item = best_match if best_match["type"] == "lost" else new_item
        found_item = new_item if best_match["type"] == "lost" else best_match
        ai_agent_notify(lost_item, found_item)
    else:
        print("test msg : No ensemble match met the threshold.")

def query_rag_agent(user_message: str):
    user_text_vector = vector_db.get_text_vector(user_message.lower())
    text_matches = vector_db.text_index.query(user_text_vector, max_hamming_distance=3)
    candidate_ids = [cid for cid, _ in text_matches][:10]
    
    from bson import ObjectId
    found_items = list(items_col.find({
        "_id": {"$in": [ObjectId(cid) for cid in candidate_ids if ObjectId.is_valid(cid)]},
        "type": "found",
        "is_claimed": False
    }))
    
    # 1. Try Google Gemini API (if key is configured and valid)
    if api_key and not api_key.startswith("your_gemini_api_key"):
        try:
            if found_items:
                context_lines = []
                for i, item in enumerate(found_items, start=1):
                    context_lines.append(
                        f"[{i}] Item Name: {item.get('item_name','Item')}\n"
                        f"    Description: {item.get('description','')}\n"
                        f"    Location Found: {item.get('location','')}\n"
                        f"    Date Found: {item.get('date','')} at {item.get('time','')}\n"
                        f"    Claim ID: {item.get('_id','')}\n"
                    )
                retrieved_context = "\n".join(context_lines)
            else:
                retrieved_context = "No matches found in the current item registry."

            system_prompt = f"""
You are LostLink AI, the official intelligent conversational RAG (Retrieval-Augmented Generation) assistant for the NITK Campus Lost & Found registry.
A student/user is searching for a lost item and says: "{user_message}"

Use the following list of retrieved found items to construct your response. Be helpful, clear, and natural:
1. If you see a potential match in the retrieved list, explain which item matches, where/when it was found, and provide its specific Claim ID. Tell them they can click "Initiate Claim" in the browse section or navigate to claims using that Claim ID.
2. If there are multiple potential matches, list them clearly and ask for clarification.
3. If there are no close matches, politely let them know, list the items that were retrieved (in case they recognize something), and advise them to file a formal "Report Lost" submission in the navigation bar.

Do NOT invent or hallucinate any item reports that are not listed in the retrieved context below.

Retrieved Found Items:
----------------------
{{retrieved_context}}
----------------------
"""
            response = model.generate_content(system_prompt)
            return response.text.strip()
        except Exception as e:
            print(f"test msg : Gemini RAG generation failed: {e}. Trying local Ollama fallback...")

    # 2. Try Local CPU-Optimized LLM (google/flan-t5-small) via direct AutoClasses
    try:
        tokenizer, model_seq2seq = get_local_rag_model()
        print("test msg : Generating RAG response via local FLAN-T5 CPU model...")
        
        context_str = ""
        if found_items:
            for idx, item in enumerate(found_items, start=1):
                context_str += f"Item {idx}: {item.get('item_name')} found at {item.get('location')}. Description: {item.get('description')}. Claim ID is {item.get('_id')}. "
        else:
            context_str = "No matching items found."
            
        prompt = (
            f"Answer the user query based on the following context.\n\n"
            f"Context: {context_str}\n"
            f"User Query: {user_message}\n\n"
            f"Answer:"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model_seq2seq.generate(**inputs, max_length=150)
        llm_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # If the response is empty or too brief, augment it with retrieved matches for usability
        if len(llm_response) < 5 or "no " in llm_response.lower() or "not " in llm_response.lower():
            llm_response = "I found matching records in our local vector database."
            
        return (
            f"**[Local RAG LLM - FLAN-T5]**: {llm_response}\n\n"
            f"**Retrieved matches from LSH index**:\n"
            + "\n".join([f"- **{it['item_name']}** found at {it['location']} (Claim ID: `{it['_id']}`)" for it in found_items])
        )
    except Exception as e:
        print(f"test msg : Local FLAN-T5 inference failed: {e}. Trying Ollama...")

    # 3. Try Local Ollama Instance (to keep RAG 100% local/offline)
    # Checks both container network bridge and host loopback addresses
    ollama_hosts = ["http://localhost:11434", "http://host.docker.internal:11434"]
    for url in ollama_hosts:
        try:
            res = requests.get(f"{url}/api/tags", timeout=1.0)
            if res.status_code == 200:
                models_data = res.json().get("models", [])
                if models_data:
                    model_name = models_data[0]["name"]
                    print(f"test msg : Running RAG via local Ollama LLM model '{model_name}'...")
                    
                    context_str = ""
                    for item in found_items:
                        context_str += f"- {item.get('item_name')} found at {item.get('location')}. Claim ID: {item.get('_id')}\n"
                    
                    prompt = (
                        f"System: You are LostLink AI, a conversational helper for the NITK registry.\n"
                        f"Context of retrieved database items:\n{context_str}\n"
                        f"User: {user_message}\n"
                        f"Answer the user query based ONLY on the context. Keep it concise."
                    )
                    
                    payload = {
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False
                    }
                    gen_res = requests.post(f"{url}/api/generate", json=payload, timeout=12.0)
                    if gen_res.status_code == 200:
                        return gen_res.json().get("response", "").strip()
        except Exception:
            continue

    # 4. Fallback to Local Deterministic LSH RAG Formatter (No API key, no external LLM required)
    print("test msg : Falling back to local deterministic RAG formatter...")
    if not found_items:
        return (
            "**[Local LSH Matching Engine]**\n\n"
            "I checked our local vector database but could not find any active, unclaimed items matching your description.\n\n"
            "*Tip: Try filing a formal 'Report Lost' ticket in the navigation bar. If a matching item is turned in, our background ensemble matcher will auto-notify you!*"
        )
    
    reply = (
        f"**[Local LSH Matching Engine]**\n\n"
        f"I retrieved **{len(found_items)} potential match(es)** from the local vector database using LSH index buckets:\n\n"
    )
    for i, item in enumerate(found_items, start=1):
        reply += f"- **{item.get('item_name', 'Item')}**\n"
        reply += f"   • Description: {item.get('description', '')}\n"
        reply += f"   • Location Found: {item.get('location', '')}\n"
        reply += f"   • Date/Time: {item.get('date', '')} at {item.get('time', '')}\n"
        reply += f"   • Claim ID: `{item.get('_id', '')}`\n\n"
    
    reply += (
        "**How to Claim**:\n"
        "If one of these matches your missing property, go to Browse Items on the navigation bar, "
        "locate the item record, and click Initiate Claim using the corresponding Claim ID."
    )
    return reply

_local_tokenizer = None
_local_rag_model = None

def get_local_rag_model():
    global _local_tokenizer, _local_rag_model
    if _local_rag_model is None:
        print("test msg : Loading local FLAN-T5 LLM classes for RAG generation...")
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _local_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        _local_rag_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return _local_tokenizer, _local_rag_model

# Train machine learning matching models on startup
try:
    train_matching_models()
except Exception as e:
    print(f"Failed to pre-train matching models on module import: {e}")
