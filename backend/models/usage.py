from sqlalchemy import Column, Integer, String, ForeignKey, Float, DateTime
from sqlalchemy.sql import func
from backend.database import Base

class Usage(Base):
    __tablename__ = "usage"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    provider = Column(String)
    tokens_used = Column(Integer)
    cost_estimate = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())