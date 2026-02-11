import os
import google.generativeai as genai
from database import items_col
from notif import send_email, speak_message, make_phone_call
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY", "").strip()
if api_key.startswith('"') and api_key.endswith('"'):
    api_key = api_key[1:-1]

print(f"📡 AI Matcher: Configuring Gemini with key starting with {api_key[:5]}...")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-flash-latest")

# --- Local ML Lazy Loading (Standard Mode) ---
_local_sim_model = None
_local_transform = None

def get_local_model():
    """Lazily loads the MobileNetV2 model to save RAM on startup."""
    global _local_sim_model, _local_transform
    if _local_sim_model is None:
        print("⚙️ Loading local MobileNetV2 for Standard similarity...")
        # MobileNetV2 is very lightweight (~14MB)
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
    """Generates a 1280-d embedding for an image using local MobileNetV2."""
    try:
        model, transform = get_local_model()
        with Image.open(image_path).convert('RGB') as img:
            img_t = transform(img).unsqueeze(0)
            with torch.no_grad():
                embedding = model(img_t)
            return embedding.squeeze().numpy()
    except Exception as e:
        print(f"❌ Error extracting local embedding: {e}")
        return None

def match_with_embeddings(new_item):
    """
    Standard Mode: Uses local embeddings to find similar images.
    """
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

    # Compare with all existing embeddings
    existing_embeddings = [np.array(item["embedding"]) for item in other_items]
    similarities = cosine_similarity([new_embedding], existing_embeddings)[0]
    
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    
    print(f"📡 Local Embedding Match Score: {best_score:.2f}")
    
    # 0.8 is a good balance for MobileNet embeddings
    if best_score > 0.82: 
        match = other_items[best_idx]
        print(f"✅ Local Visual Match Found: {match['item_name']}")
        ai_agent_notify(match, new_item)

def match_with_gemini(new_item):
    # Find opposite type (lost ↔ found) and unclaimed
    other_items = items_col.find({
        "type": {"$ne": new_item["type"]},
        "is_claimed": False
    })

    matched_contacts = set()  # To avoid duplicate alerts

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
            print(f"❌ Error from Gemini model: {e}")
            continue

        if decision.startswith("yes"):
            if existing["contact_info"] in matched_contacts:
                continue  # avoid duplicate emails/calls

            print("✅ Match found")
            ai_agent_notify(existing, new_item)
            matched_contacts.add(existing["contact_info"])
            break  # comment this line if you want to allow multiple matches

import re

def is_valid_email(address):
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", address))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def match_with_tfidf(new_item):
    """
    Enhanced TF-IDF matching that considers description, location, and time.
    """
    target_type = "found" if new_item["type"] == "lost" else "lost"
    all_items = list(items_col.find({"type": target_type, "is_claimed": False}))
    
    if not all_items:
        print(f"No {target_type} items to match against.")
        return

    # Combine description and location for richer text matching
    def get_text(it):
        return f"{it.get('description', '')} {it.get('location', '')}".lower()

    corpus = [get_text(item) for item in all_items]
    new_text = get_text(new_item)
    corpus.append(new_text)

    # TF-IDF Vectorization with word and character n-grams for better matching on short/noisy text
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3), 
        analyzer='word',
        stop_words='english',
        min_df=1
    ).fit_transform(corpus)
    vectors = vectorizer.toarray()

    # Calculate text similarity
    similarities = cosine_similarity([vectors[-1]], vectors[:-1])[0]

    # Add Time-based logic and Location bonus
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
        
        # Location Bonus (if exact match in string)
        new_loc = new_item.get("location", "").lower()
        old_loc = item.get("location", "").lower()
        if new_loc and old_loc and (new_loc in old_loc or old_loc in new_loc):
            score += 0.2  # Increased bonus for location match
        
        # Time Logic: Extra score if time lost < time found
        item_dt = parse_dt(item["date"], item["time"])
        if new_dt and item_dt:
            if new_item["type"] == "lost" and new_dt <= item_dt:
                score += 0.15 # Lost before found is logical
            elif new_item["type"] == "found" and item_dt <= new_dt:
                score += 0.15 # Found after lost is logical
        
        scored_matches.append((score, item))

    # Sort by score
    scored_matches.sort(key=lambda x: x[0], reverse=True)
    best_score, best_match = scored_matches[0]

    print(f"🔍 Enhanced Match (TF-IDF+Context): {best_match['item_name']} (Final Score: {best_score:.2f})")

    # Threshold 0.75 is better for character n-grams combinations
    if best_score > 0.75:
        ai_agent_notify(best_match if best_match["type"] == "lost" else new_item, 
                        new_item if best_match["type"] == "lost" else best_match)
    else:
        print("❌ No strong match found.")


import qrcode
from io import BytesIO
import base64

def generate_qr_for_item(item_id: str):
    # Detect environment for QR link (could be improved with dynamic detection)
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
    subject = "🎯 Possible Match for Your Lost Item!"
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    body = f"""
Hi,

We may have found a match for your lost item: {lost_item['item_name']}.

Matched with found item:
- {found_item['item_name']}
- Description: {found_item['description']}
- Location: {found_item['location']}
- Date: {found_item['date']} at {found_item['time']}

Please log in to LostLink AI to view details and confirm:
{base_url}/login.html

– LostLink AI Agent 🤖

DO NOT REPLY. 
THIS IS AUTO-GENERATED.
"""

    print(f"📧 Notifying {contact} about match with {found_item['item_name']}")

    # Send email only if the contact is an email
    if is_valid_email(contact):
        send_email(to=contact, subject=subject, body=body)
        print(f"✅ Email sent to {contact}")
    else:
        print(f"⚠️ Skipping email: {contact} is not a valid email.")

    speak_message(
        f"A potential match has been found for lost item: {lost_item['item_name']} at {found_item['location']}."
    )

    if lost_item.get("wants_call") and contact.startswith("+"):
        make_phone_call(
            to_number=contact,
            message=f"Hello! A match is found for your lost item: {lost_item['item_name']} at {found_item['location']}. Please check LostLink AI. Thank you."
        )

def generate_image_description(image_path):
    """
    Uses Gemini to generate a description of an item from an image.
    """
    try:
        # Load the image
        from PIL import Image
        img = Image.open(image_path)
        
        try:
            # Prompt for Gemini
            prompt = "Describe this object in detail for a lost and found system. Focus on color, brand, unique markings, and material. Keep it concise but descriptive (max 100 words)."
            
            # Generate content
            response = model.generate_content([prompt, img])
            return response.text.strip()
        finally:
            img.close()  # Ensure file is closed even if Gemini fails
    except Exception as e:
        import traceback
        print(f"❌ Error generating image description: {e}")
        traceback.print_exc()
        return None

def generate_local_description(image_path):
    """
    Standard Mode: Generates a basic description using local ImageNet tags.
    This is extremely fast and light.
    """
    try:
        from torchvision.models import MobileNet_V2_Weights
        weights = MobileNet_V2_Weights.DEFAULT
        # Get the categories
        categories = weights.meta["categories"]
        
        # We'll use the classification model for this one
        model, transform = get_local_model()
        # Note: local_sim_model (Identify head) is for embeddings. 
        # For actual classification we need the full model. 
        # I will load a separate classifier lazily.
        classifier = _get_local_classifier()
        
        with Image.open(image_path).convert('RGB') as img:
            img_t = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = classifier(img_t)
            
            # Get top prediction only
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            prob, catid = torch.max(probabilities, 0)
            
            return f"Identified as: {categories[catid.item()]}"
    except Exception as e:
        print(f"❌ Local description failed: {e}")
        return "Local description unavailable."

_local_classifier = None
def _get_local_classifier():
    global _local_classifier
    if _local_classifier is None:
        weights = MobileNet_V2_Weights.DEFAULT
        _local_classifier = mobilenet_v2(weights=weights).eval()
    return _local_classifier
