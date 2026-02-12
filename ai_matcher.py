import os
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
            return embedding.squeeze().numpy()
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
        "embedding": {"$exists": True}
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

    def parse_dt(d, t):
        try:
            return datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M")
        except:
            return None

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
