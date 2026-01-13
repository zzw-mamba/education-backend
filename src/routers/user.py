from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import or_
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from database import get_db
from models import User

router = APIRouter(prefix="/auth", tags=["User"])

# 密码哈希配置
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# Pydantic 模型
class UserCreate(BaseModel):
    username: str
    password: str
    email: EmailStr | None = None

class UserLogin(BaseModel):
    username: str
    password: str

class UserUpdate(BaseModel):
    user_id: int
    email: EmailStr | None = None

class ChangePassword(BaseModel):
    user_id: int
    old_password: str
    new_password: str

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(user: UserCreate, db: Session = Depends(get_db)):
    # 检查用户名是否已存在
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # 创建新用户
    hashed_password = get_password_hash(user.password)
    new_user = User(
        username=user.username,
        password_hash=hashed_password,
        email=user.email
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"id": new_user.id, "username": new_user.username, "email": new_user.email}

@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(
        or_(
            User.username == user.username,
            User.email == user.username
        )
    ).first()
    
    if not db_user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    if not verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    # 实际项目中应返回 JWT Token，这里仅返回用户信息
    return {
        "message": "Login successful",
        "user": {
            "id": db_user.id,
            "username": db_user.username,
            "email": db_user.email
        }
    }

@router.put("/update", status_code=status.HTTP_200_OK)
def update_user_info(user_update: UserUpdate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_update.user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user_update.email is not None:
        db_user.email = user_update.email
    
    db.commit()
    db.refresh(db_user)
    return {"id": db_user.id, "username": db_user.username, "email": db_user.email}

@router.post("/change-password", status_code=status.HTTP_200_OK)
def change_password(data: ChangePassword, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == data.user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if not verify_password(data.old_password, db_user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect old password")
    
    if len(data.new_password) < 6:
         raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    db_user.password_hash = get_password_hash(data.new_password)
    db.commit()
    return {"message": "Password updated successfully"}
