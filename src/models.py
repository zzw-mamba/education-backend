from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True, comment="用户ID")
    username = Column(String(50), unique=True, index=True, nullable=False, comment="用户名")
    password_hash = Column(String(255), nullable=False, comment="密码哈希")
    email = Column(String(100), unique=True, index=True, nullable=True, comment="邮箱")
    
    # 状态：active, inactive 等
    status = Column(String(20), default="active", comment="账户状态")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")

    logs = relationship("Log", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True, comment="知识条目ID")
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="所属用户ID")
    
    title = Column(String(200), nullable=False, index=True, comment="标题")
    content = Column(Text, nullable=True, comment="内容文本")
    category = Column(String(100), nullable=True, comment="分类标签")

    # 如果是文件上传，可以存储文件路径
    file_path = Column(String(500), nullable=True, comment="源文件路径")
    file_type = Column(String(50), nullable=True, comment="文件类型 (pdf, docx, md)")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")

    def __repr__(self):
        return f"<KnowledgeBase(id={self.id}, title='{self.title}')>"


class Log(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True, comment="日志ID")
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, comment="关联用户ID(可为空)")
    
    action = Column(String(100), nullable=False, comment="操作类型 (如: login, upload_file, query)")
    details = Column(Text, nullable=True, comment="操作详情(JSON或文本)")
    ip_address = Column(String(50), nullable=True, comment="IP地址")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="记录时间")

    # 关系定义
    user = relationship("User", back_populates="logs")

    def __repr__(self):
        return f"<Log(id={self.id}, action='{self.action}')>"
