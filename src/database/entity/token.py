from sqlalchemy import Column, Integer, Boolean, String, DateTime
from src.database.entity.base import Base

# API TOKEN
class Token(Base):
    __tablename__ = 't_tokens'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    token = Column(String, unique=True, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    is_admin = Column(Boolean, nullable=False, default=False)
    last_used = Column(DateTime, nullable=True)
