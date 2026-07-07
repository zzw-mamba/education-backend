"""数据库模型模块

定义 SQLAlchemy ORM 模型类，映射数据库表结构。

包含的主要模型：
- User: 用户模型
- Log: 生成日志模型
- Template: 模板模型
- Tag: 标签模型
- KnowledgeBase: 知识库模型
- KBTagRelation: 知识库-标签关联模型
- KBService: 知识库服务类（包含全文检索和推荐功能）
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index, text
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func
from database import Base
import jieba.analyse
from sqlalchemy.dialects.mysql import LONGTEXT


class User(Base):
    """用户模型。
    
    存储系统用户信息，包括用户名、密码哈希、邮箱等。
    
    属性：
        id: 用户ID（主键）
        username: 用户名（唯一）
        password_hash: 密码哈希
        email: 邮箱（唯一）
        status: 账户状态（active/inactive）
        created_at: 创建时间
        updated_at: 更新时间
        logs: 关联的日志记录
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True, comment="用户ID")
    username = Column(String(50), unique=True, index=True, nullable=False, comment="用户名")
    password_hash = Column(String(255), nullable=False, comment="密码哈希")
    email = Column(String(100), unique=True, index=True, nullable=True, comment="邮箱")
    
    status = Column(String(20), default="active", comment="账户状态")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")

    logs = relationship("Log", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"


class Log(Base):
    """生成日志模型。
    
    记录用户使用模板生成内容的历史记录。
    
    属性：
        id: 日志ID（主键）
        user_id: 关联用户ID
        template_id: 关联模板ID
        knowledge_ids: 关联知识库ID列表（逗号分隔）
        result_path: 生成结果文件路径
        created_at: 记录时间
        user: 关联的用户对象
        template: 关联的模板对象
    """
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True, comment="日志ID")
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, comment="关联用户ID(可为空)")
    
    template_id = Column(Integer, ForeignKey("templates.id"), comment="关联模板ID")
    knowledge_ids = Column(String(255), nullable=False, comment="关联知识库ID列表，逗号分隔")
    result_path = Column(Text, nullable=True, comment="生成结果")

    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="记录时间")

    user = relationship("User", back_populates="logs")
    template = relationship("Template", back_populates="logs")

    def __repr__(self):
        return f"<Log(id={self.id}, template_id={self.template_id})>"


class Template(Base):
    """模板模型。
    
    存储用户创建的文本生成模板，包含提示词、名称、类别等信息。
    
    属性：
        id: 模板ID（主键）
        user_id: 用户ID（外键）
        icon_path: 图标文件路径
        prompt: 提示词
        name: 模板名称
        category: 模板类别（0:通用, 1:科技, 2:医疗, 3:管理, 4:论文）
        description: 模板描述
        example: 示例
        labels: 标签（逗号分隔）
        created_at: 创建时间
        updated_at: 更新时间
        user: 关联的用户对象
        logs: 关联的日志记录
    """
    __tablename__ = "templates"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True, comment="模板ID")
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="用户ID")
    icon_path = Column(String(100), nullable=True, comment="图标文件路径")
    prompt = Column(Text, nullable=False, comment="提示词")
    name = Column(String(50), nullable=False, comment="模板名称")
    category = Column(Integer, nullable=True, default=0, comment="模板类别")
    description = Column(Text, nullable=True, comment="模板描述")
    example = Column(Text, nullable=True, comment="示例")
    labels = Column(String(200), nullable=True, comment="标签，逗号分隔")

    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")

    user = relationship("User", backref="templates")
    logs = relationship("Log", back_populates="template")

    def __repr__(self):
        return f"<Template(id={self.id}, user_id={self.user_id})>"


class KBTagRelation(Base):
    """知识库-标签关联模型。
    
    实现知识库与标签之间的多对多关系。
    
    属性：
        kb_id: 知识库ID（主键）
        tag_id: 标签ID（主键）
    """
    __tablename__ = "kb_tag_relation"
    kb_id = Column(Integer, ForeignKey("knowledge_base.id"), primary_key=True)
    tag_id = Column(Integer, ForeignKey("tags.id"), primary_key=True)


class Tag(Base):
    """标签模型。
    
    存储知识标签，用于知识库内容的分类和检索。
    
    属性：
        id: 标签ID（主键）
        name: 标签名称（唯一）
        kb_items: 关联的知识库条目
    """
    __tablename__ = "tags"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, index=True, nullable=False)
    
    kb_items = relationship("KnowledgeBase", secondary="kb_tag_relation", back_populates="tags")


class KnowledgeBase(Base):
    """知识库模型。
    
    存储学术论文、文档等知识内容，支持全文检索。
    
    属性：
        id: 知识库ID（主键）
        title: 标题
        content: 内容（长文本）
        category: 类别
        authors: 作者
        file_path: 文件路径
        file_type: 文件类型
        year: 年份
        created_at: 创建时间
        updated_at: 更新时间
        tags: 关联的标签列表
    """
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

    tags = relationship("Tag", secondary="kb_tag_relation", back_populates="kb_items")

    __table_args__ = (
        Index('ix_fulltext_title_content', 'title', 'content', mysql_prefix='FULLTEXT', mysql_with_parser='ngram'),
        Index('ix_fulltext_title', 'title', mysql_prefix='FULLTEXT', mysql_with_parser='ngram'),
        Index('ix_fulltext_content', 'content', mysql_prefix='FULLTEXT', mysql_with_parser='ngram'),
    )


class KBService:
    """知识库服务类。
    
    提供知识库的增删改查、全文检索和推荐功能。
    
    主要方法：
        add_entry: 添加知识条目并自动提取关键词建立标签
        search: 全文检索（基于 MySQL MATCH AGAINST）
        recommend_similar: 基于共同标签推荐相似条目
    """

    @staticmethod
    def add_entry(db: Session, title: str, content: str, category: str = None):
        """新增知识条目，并自动提取关键词建立标签。
        
        使用 jieba 分词工具从标题和内容中提取关键词，自动创建或关联标签。
        
        Args:
            db: 数据库会话
            title: 知识条目标题
            content: 知识条目内容
            category: 类别（可选）
            
        Returns:
            KnowledgeBase: 创建的知识条目对象
        """
        new_entry = KnowledgeBase(
            title=title,
            content=content,
            category=category
        )
        db.add(new_entry)
        db.flush()

        text_to_analyze = f"{title} {title} {content}" 
        keywords = jieba.analyse.extract_tags(text_to_analyze, topK=5)

        for kw in keywords:
            tag = db.query(Tag).filter(Tag.name == kw).first()
            if not tag:
                tag = Tag(name=kw)
                db.add(tag)
                db.flush()
            new_entry.tags.append(tag)
        
        db.commit()
        db.refresh(new_entry)
        return new_entry

    @staticmethod
    def search(db: Session, keyword: str, limit: int = 10):
        """全文检索：基于 MySQL MATCH AGAINST。
        
        使用 MySQL 全文索引进行自然语言模式检索，返回匹配结果和相似度分数。
        
        Args:
            db: 数据库会话
            keyword: 检索关键词
            limit: 返回结果数量限制（默认10）
            
        Returns:
            list: 包含 (id, title, score) 的结果列表
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
        """推荐系统：基于共同标签（标签重合度）。
        
        根据知识条目之间的标签重合度推荐相似的知识条目。
        
        Args:
            db: 数据库会话
            kb_id: 目标知识条目ID
            limit: 返回推荐数量限制（默认5）
            
        Returns:
            list: 包含 (id, title, authors, year, common_tags) 的推荐列表
        """
        recommend_sql = text("""
            SELECT r2.kb_id AS id, k.title, k.authors, k.year, COUNT(*) as common_tags
            FROM kb_tag_relation r1
            JOIN kb_tag_relation r2 ON r1.tag_id = r2.tag_id
            JOIN knowledge_base k ON r2.kb_id = k.id
            WHERE r1.kb_id = :target_id AND r2.kb_id <> :target_id
            GROUP BY r2.kb_id, k.title, k.authors, k.year
            ORDER BY common_tags DESC
            LIMIT :limit
        """)
        return db.execute(recommend_sql, {"target_id": kb_id, "limit": limit}).all()