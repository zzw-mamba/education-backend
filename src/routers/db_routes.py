from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import get_db

router = APIRouter(tags=["Database"])

@router.get("/db-test")
def test_db_connection(db: Session = Depends(get_db)):
    """测试数据库连接"""
    try:
        db.execute(text("SELECT 1"))
        return {"status": "success", "message": "Database connection established!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
