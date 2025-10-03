from pydantic import BaseModel
from datetime import datetime

class MeetingBase(BaseModel):
    pass
class MeetingCreate(MeetingBase):
    pass

class MeetingResponse(MeetingBase):
    id : int

    class config:
        from_attribute : True # Enable ORM support for SQLAlchemy


