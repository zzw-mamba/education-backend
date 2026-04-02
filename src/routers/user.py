from fastapi import APIRouter, Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import or_
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from database import get_db
from models import User

router = APIRouter(prefix="/auth", tags=["User"])

# JWT 配置
SECRET_KEY = "your-secret-key-here-change-in-production"  # 生产环境应从环境变量读取
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 密码哈希配置
# 当前环境里 bcrypt 5.x 与 passlib 1.7.4 存在兼容性问题，主方案改用 argon2；
# 同时保留旧 bcrypt / bcrypt_sha256 哈希的验证能力。
pwd_context = CryptContext(schemes=["argon2", "bcrypt_sha256", "bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# JWT 验证依赖
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    """
    验证 JWT Token 并返回当前用户
    用法：current_user = Depends(get_current_user)
    """
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    db_user = db.query(User).filter(User.id == int(user_id)).first()
    if db_user is None:
        raise credentials_exception
    
    return db_user

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

@router.post("/login", status_code=status.HTTP_200_OK)
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
    
    # 创建 JWT Token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(db_user.id), "username": db_user.username},
        expires_delta=access_token_expires
    )
    
    return {
        "status": 200,
        "access_token": access_token,
        "token_type": "bearer",
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

@router.get("/me", status_code=status.HTTP_200_OK)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    获取当前登录用户信息
    需要在请求头中携带: Authorization: Bearer <token>
    """
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email
    }
