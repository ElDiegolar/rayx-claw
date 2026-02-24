from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from auth import token_manager
from config import Settings
from models import Agent, UserMessage, WSMessage
from orchestrator import Orchestrator
from tools import _voicebox_health

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger(__name__)

settings = Settings()
app = FastAPI(title="Multi-Agent Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/persona")
async def get_persona():
    return {"name": settings.persona_name}


@app.post("/api/persona")
async def set_persona(body: dict):
    name = body.get("name", "").strip()
    if name:
        settings.persona_name = name
    return {"name": settings.persona_name}


@app.get("/api/token")
async def get_token_info():
    """Return OAuth token status for UI timer."""
    return token_manager.get_info()


@app.post("/api/token/refresh")
async def refresh_token():
    """Force token refresh."""
    await token_manager._refresh_async()
    return token_manager.get_info()


@app.get("/api/status")
async def tool_status():
    """Return health status of all tools for UI indicators."""
    import httpx as _httpx

    statuses = {
        "filesystem": {"ok": True, "label": "Filesystem", "desc": "Read, write, search files in workspace"},
        "memory": {"ok": True, "label": "Memory", "desc": "Persistent key-value memory across sessions"},
    }

    # Check Voicebox
    try:
        statuses["voicebox"] = {
            "ok": _voicebox_health(),
            "label": "Voicebox",
            "desc": "Sophia's cloned voice (TTS via Voicebox server)",
        }
    except Exception:
        statuses["voicebox"] = {"ok": False, "label": "Voicebox", "desc": "Sophia's cloned voice (TTS via Voicebox server)"}

    # Check MiniMax API
    try:
        async with _httpx.AsyncClient(timeout=5) as client:
            r = await client.get("https://api.minimax.io/anthropic")
            statuses["minimax"] = {
                "ok": r.status_code < 500,
                "label": "MiniMax",
                "desc": "MiniMax sub-agents for parallel task delegation",
            }
    except Exception:
        statuses["minimax"] = {"ok": False, "label": "MiniMax", "desc": "MiniMax sub-agents for parallel task delegation"}

    # Check Claude API (OAuth) — uses token_manager for fresh token
    try:
        token = await token_manager.get_token_async()
        async with _httpx.AsyncClient(timeout=5) as client:
            r = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "Authorization": f"Bearer {token}",
                    "anthropic-beta": "oauth-2025-04-20",
                    "anthropic-version": "2023-06-01",
                },
            )
            statuses["claude"] = {
                "ok": r.status_code == 200,
                "label": "Claude",
                "desc": "Claude Opus — main orchestrator brain",
            }
    except Exception:
        statuses["claude"] = {"ok": False, "label": "Claude", "desc": "Claude Opus — main orchestrator brain"}

    return statuses


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    import asyncio as _aio

    await ws.accept()
    orchestrator = Orchestrator()
    log.info("Client connected")

    async def send_ws(msg: WSMessage):
        await ws.send_text(msg.model_dump_json())

    # Send chat history to client for replay
    ui_history = orchestrator.get_ui_history()
    if ui_history:
        log.info("Sending %d history exchanges to client", len(ui_history))
        await ws.send_text(WSMessage(
            type="history",
            agent=Agent.SYSTEM,
            history=ui_history,
        ).model_dump_json())

    running_task: _aio.Task | None = None

    try:
        while True:
            raw = await ws.receive_text()
            user_msg = UserMessage(**json.loads(raw))

            if user_msg.type == "cancel":
                if running_task and not running_task.done():
                    log.info("Cancel requested")
                    orchestrator.cancel()
                    running_task.cancel()
                    try:
                        await running_task
                    except (_aio.CancelledError, Exception):
                        pass
                    running_task = None
                continue

            if user_msg.type == "message" and user_msg.content.strip():
                log.info("Task received: %s", user_msg.content[:80])
                running_task = _aio.create_task(
                    orchestrator.run(user_msg.content, send=send_ws)
                )

    except WebSocketDisconnect:
        log.info("Client disconnected")
        if running_task and not running_task.done():
            running_task.cancel()


# Static files mounted last so /ws takes priority
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
