from sqlalchemy import Column, Integer, Text, DateTime, String, ForeignKey, Interval
from database import Base
from datetime import datetime

class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    date = Column(DateTime, nullable=False)
    transcript_id = Column(Integer, ForeignKey("transcriptions.id"), nullable=False)
    summary = Column(Text)
    participants = Column(Text)
    duration = Column(Interval, default=None)
    key_points = Column(Text)
    action_items = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

class Transcription(Base):
    __tablename__ = "transcriptions"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, nullable=True)
    transcript = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)