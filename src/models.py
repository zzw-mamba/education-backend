from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index, text
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func
from database import Base
import jieba.analyse
from sqlalchemy.dialects.mysql import LONGTEXT

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
    
    
class KBTagRelation(Base):
    __tablename__ = "kb_tag_relation"
    kb_id = Column(Integer, ForeignKey("knowledge_base.id"), primary_key=True)
    tag_id = Column(Integer, ForeignKey("tags.id"), primary_key=True)

class Tag(Base):
    __tablename__ = "tags"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, index=True, nullable=False)
    
    kb_items = relationship("KnowledgeBase", secondary="kb_tag_relation", back_populates="tags")

class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    title = Column(String(200), nullable=False)
    content = Column(LONGTEXT, nullable=True)
    category = Column(String(100), nullable=True)
    authors = Column(Text, nullable=True)
    file_path = Column(String(500), nullable=True)
    file_type = Column(String(50), nullable=True)
    year = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关联
    tags = relationship("Tag", secondary="kb_tag_relation", back_populates="kb_items")

    # --- 核心：MySQL 全文索引 (针对搜索) ---
    __table_args__ = (
        Index('ix_fulltext_title_content', 'title', 'content', mysql_prefix='FULLTEXT', mysql_with_parser='ngram'),
        # 独立索引：用于给标题额外加权
        Index('ix_fulltext_title', 'title', mysql_prefix='FULLTEXT', mysql_with_parser='ngram'),
        # 独立索引：用于内容搜索
        Index('ix_fulltext_content', 'content', mysql_prefix='FULLTEXT', mysql_with_parser='ngram'),
    )


class KBService:
    @staticmethod
    def add_entry(db: Session, title: str, content: str, category: str = None):
        """
        新增知识条目，并自动提取关键词建立标签
        """
        # 1. 创建基础条目
        new_entry = KnowledgeBase(
            title=title,
            content=content,
            category=category
        )
        db.add(new_entry)
        db.flush()  # 获取 id

        # 2. 自动提取关键词 (提取前 5 个)
        # 权重：标题权重更高，所以拼接在一起处理
        text_to_analyze = f"{title} {title} {content}" 
        keywords = jieba.analyse.extract_tags(text_to_analyze, topK=5)

        # 3. 维护标签关系
        for kw in keywords:
            # 查找标签是否存在，不存在则创建
            tag = db.query(Tag).filter(Tag.name == kw).first()
            if not tag:
                tag = Tag(name=kw)
                db.add(tag)
                db.flush()
            
            # 建立多对多关联
            new_entry.tags.append(tag)
        
        db.commit()
        db.refresh(new_entry)
        return new_entry

    @staticmethod
    def search(db: Session, keyword: str, limit: int = 10):
        """
        全文检索：基于 MySQL MATCH AGAINST
        """
        query_sql = text("""
            SELECT id, title, MATCH(title, content) AGAINST(:kw IN NATURAL LANGUAGE MODE) AS score
            FROM knowledge_base
            WHERE MATCH(title, content) AGAINST(:kw IN NATURAL LANGUAGE MODE)
            ORDER BY score DESC
            LIMIT :limit
        """)
        result = db.execute(query_sql, {"kw": keyword, "limit": limit}).all()
        return result

    @staticmethod
    def recommend_similar(db: Session, kb_id: int, limit: int = 5):
        """
        推荐系统：基于共同标签（标签重合度）
        """
        recommend_sql = text("""
            SELECT r2.kb_id, COUNT(*) as common_tags
            FROM kb_tag_relation r1
            JOIN kb_tag_relation r2 ON r1.tag_id = r2.tag_id
            WHERE r1.kb_id = :target_id AND r2.kb_id <> :target_id
            GROUP BY r2.kb_id
            ORDER BY common_tags DESC
            LIMIT :limit
        """)
        return db.execute(recommend_sql, {"target_id": kb_id, "limit": limit}).all()
