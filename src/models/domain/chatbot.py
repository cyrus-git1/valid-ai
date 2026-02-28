from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
import Optional
from src.utils.time import utcnow
from pydantic import Field
from ._time import utcnow

class ChatBotTranscript(BaseModel):
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None

    transcript_uri: str

    created_at: datetime = Field(default_factory=utcnow)
    ended_at: Optional[datetime] = None

class ChatBotRow(ChatBotTranscript):
    id: UUID
    created_at: datetime
    ended_at: datetime
