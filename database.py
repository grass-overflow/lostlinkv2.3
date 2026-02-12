from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017").strip()
if MONGO_URI.startswith('"') and MONGO_URI.endswith('"'):
    MONGO_URI = MONGO_URI[1:-1]

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

try:
    client.admin.command('ping')
    print(f"Connected to MongoDB at {MONGO_URI.split('@')[-1] if '@' in MONGO_URI else MONGO_URI}")
except Exception as e:
    print(f"Error: Could not connect to MongoDB.")
    print(f"Attempted URI: {MONGO_URI}")
    print(f"Tip: If using Atlas, check your IP whitelist. If local, ensure MongoDB is running.")
    print(f"Technical details: {e}")

db = client["lostlink_ai"]
items_col = db["items"]
users_col = db["users"]
feedback_col = db["feedback"]
matches_col = db["matches"]
