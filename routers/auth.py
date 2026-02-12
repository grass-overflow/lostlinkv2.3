from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from database import users_col
from jose import jwt, JWTError
from datetime import datetime, timedelta
import os
from models import UserSignup, UserLogin
from passlib.context import CryptContext

router = APIRouter(tags=["auth"])

pwd_context = CryptContext(schemes=["bcrypt_sha256", "bcrypt"], deprecated="auto")


SECRET = os.getenv("JWT_SECRET", "secret123")
ALGORITHM = "HS256"
TOKEN_EXPIRE_MIN = 60 * 24

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

security = HTTPBearer()

def create_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MIN)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"email": email}

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/signup")
def signup(user: UserSignup):
    if users_col.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_dict = user.model_dump()
    user_dict["password"] = get_password_hash(user.password)
    users_col.insert_one(user_dict)
    return {"message": "User created successfully"}

@router.post("/login")
def login(user_credentials: UserLogin):
    found = users_col.find_one({"email": user_credentials.email})
    if not found or not verify_password(user_credentials.password, found["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token({"sub": user_credentials.email})
    return {"access_token": token}
@router.get("/me")
def get_user_info(user: dict = Depends(get_current_user)):
    db_user = users_col.find_one({"email": user["email"]})
    if not db_user:
         raise HTTPException(status_code=404, detail="User not found")
    return {"name": db_user.get("name", "User"), "email": db_user.get("email")}
