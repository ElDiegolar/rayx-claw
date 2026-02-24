from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel


class Agent(str, Enum):
    CLAUDE = "claude"
    MINIMAX = "minimax"
    SYSTEM = "system"


class WSMessage(BaseModel):
    """Server -> Client WebSocket message."""
    type: Literal["chunk", "status", "error", "done", "tool_use", "tool_result", "history"]
    agent: Agent
    content: str = ""
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    history: Optional[list] = None


class UserMessage(BaseModel):
    """Client -> Server WebSocket message."""
    type: Literal["message", "cancel"]
    content: str = ""
