from pydantic import BaseModel
from typing import List, Optional


class ChatMessage(BaseModel):
    role: str
    content: str


# ADD THIS
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    repo_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    citations: List[str]
    tool_calls_made: List[str]
    confidence: float
    next_steps_suggestion: Optional[str] = None


class ChatHistory(BaseModel):
    messages: List[ChatMessage]