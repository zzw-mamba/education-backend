from database import engine, Base
import models

def init_db():
    print("Creating database tables...")
    # Base.metadata.create_all 将会创建所有继承自 Base 的模型对应的表
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully!")

if __name__ == "__main__":
    init_db()
