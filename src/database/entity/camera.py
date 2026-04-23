from sqlalchemy import Column, Integer, String, Boolean, JSON
from src.database.entity.base import Base

class Camera(Base):
    __tablename__ = 'm_cameras'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    url = Column(String, nullable=False)
    is_enabled = Column(Boolean, nullable=False, default=False)
    detect_fps = Column(Integer, nullable=False, default=5)
    snapshot_enabled = Column(Boolean, nullable=False, default=True)
    mqtt_enabled = Column(Boolean, nullable=False, default=True)
