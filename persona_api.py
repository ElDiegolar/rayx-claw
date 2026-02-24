"""Persona management API — CRUD endpoints for persona YAML files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException

from config import Settings
from persona import Persona, load_persona, PERSONAS_DIR, _deep_merge

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/personas", tags=["personas"])


def _list_persona_files() -> list[str]:
    """Return list of persona names (filenames without .yaml)."""
    if not PERSONAS_DIR.exists():
        return []
    return sorted(
        p.stem for p in PERSONAS_DIR.glob("*.yaml")
    )


def _read_yaml(name: str) -> dict[str, Any]:
    """Read a persona YAML file and return raw dict."""
    path = PERSONAS_DIR / f"{name}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Persona '{name}' not found")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(name: str, data: dict[str, Any]) -> None:
    """Write persona data to a YAML file."""
    PERSONAS_DIR.mkdir(parents=True, exist_ok=True)
    path = PERSONAS_DIR / f"{name}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _persona_summary(name: str, data: dict[str, Any]) -> dict[str, Any]:
    """Return a summary dict for list views."""
    p = Persona(data)
    return {
        "id": name,
        "name": p.name,
        "tagline": p.tagline,
        "greeting": p.greeting,
        "avatar_emoji": p.avatar_emoji,
        "primary_color": p.primary_color,
        "theme": p.theme,
        "voice_enabled": p.voice_enabled,
    }


def _persona_full(name: str, data: dict[str, Any]) -> dict[str, Any]:
    """Return full persona data for editing."""
    p = Persona(data)
    return {
        "id": name,
        "name": p.name,
        "tagline": p.tagline,
        "greeting": p.greeting,
        "personality": {
            "tone": p.tone,
            "style": p.style,
            "traits": p.traits,
        },
        "role": {
            "description": p.role_description,
            "instructions": p.instructions,
        },
        "guardrails": {
            "do": p.guardrails_do,
            "dont": p.guardrails_dont,
        },
        "voice": p.voice_config,
        "ui": {
            "avatar_emoji": p.avatar_emoji,
            "primary_color": p.primary_color,
            "theme": p.theme,
        },
    }


# ---- Active persona state ----
_active_persona_name: str = "default"
_active_persona: Persona | None = None


def get_active_persona() -> Persona:
    """Get the currently active persona (lazy-loads default)."""
    global _active_persona, _active_persona_name
    if _active_persona is None:
        _active_persona = load_persona(_active_persona_name)
    return _active_persona


def switch_active_persona(name: str) -> Persona:
    """Switch to a different persona by name."""
    global _active_persona, _active_persona_name
    # Validate it exists
    path = PERSONAS_DIR / f"{name}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Persona '{name}' not found")
    _active_persona_name = name
    _active_persona = load_persona(name)
    # Also update settings so orchestrator uses correct persona
    settings = Settings()
    settings.persona_name = name
    log.info("Switched active persona to: %s (%s)", name, _active_persona.name)
    return _active_persona


# ---- Endpoints ----

@router.get("/active")
async def get_active():
    """Get the currently active persona."""
    p = get_active_persona()
    return {
        "id": _active_persona_name,
        **p.to_api_dict(),
    }


@router.post("/active")
async def set_active(body: dict):
    """Switch the active persona. Body: { "id": "persona-filename" }"""
    persona_id = body.get("id", "").strip()
    if not persona_id:
        raise HTTPException(status_code=400, detail="Missing 'id' field")
    p = switch_active_persona(persona_id)
    return {
        "id": persona_id,
        **p.to_api_dict(),
    }


@router.get("")
async def list_personas():
    """List all available personas with summaries."""
    names = _list_persona_files()
    result = []
    for name in names:
        try:
            data = _read_yaml(name)
            summary = _persona_summary(name, data)
            summary["active"] = (name == _active_persona_name)
            result.append(summary)
        except Exception as e:
            log.warning("Failed to load persona '%s': %s", name, e)
    return result


@router.get("/{persona_id}")
async def get_persona(persona_id: str):
    """Get full persona details for editing."""
    data = _read_yaml(persona_id)
    return _persona_full(persona_id, data)


@router.post("")
async def create_persona(body: dict):
    """Create a new persona. Body must include 'id' (filename) and persona fields."""
    persona_id = body.pop("id", "").strip().lower().replace(" ", "-")
    if not persona_id:
        raise HTTPException(status_code=400, detail="Missing 'id' field")
    if persona_id == "active":
        raise HTTPException(status_code=400, detail="Reserved name")
    
    path = PERSONAS_DIR / f"{persona_id}.yaml"
    if path.exists():
        raise HTTPException(status_code=409, detail=f"Persona '{persona_id}' already exists")
    
    _write_yaml(persona_id, body)
    log.info("Created persona: %s", persona_id)
    return _persona_full(persona_id, body)


@router.put("/{persona_id}")
async def update_persona(persona_id: str, body: dict):
    """Update an existing persona."""
    path = PERSONAS_DIR / f"{persona_id}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found")
    
    # Remove 'id' from body if present (it's in the URL)
    body.pop("id", None)
    _write_yaml(persona_id, body)
    log.info("Updated persona: %s", persona_id)
    
    # If this is the active persona, reload it
    global _active_persona
    if persona_id == _active_persona_name:
        _active_persona = load_persona(persona_id)
    
    return _persona_full(persona_id, body)


@router.delete("/{persona_id}")
async def delete_persona(persona_id: str):
    """Delete a persona. Cannot delete 'default'."""
    if persona_id == "default":
        raise HTTPException(status_code=400, detail="Cannot delete the default persona")
    
    path = PERSONAS_DIR / f"{persona_id}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found")
    
    path.unlink()
    log.info("Deleted persona: %s", persona_id)
    
    # If deleted persona was active, switch to default
    global _active_persona_name, _active_persona
    if persona_id == _active_persona_name:
        _active_persona_name = "default"
        _active_persona = load_persona("default")
    
    return {"deleted": persona_id}


@router.post("/{persona_id}/duplicate")
async def duplicate_persona(persona_id: str, body: dict = None):
    """Duplicate a persona. Optional body: { "new_id": "name" }"""
    data = _read_yaml(persona_id)
    
    new_id = (body or {}).get("new_id", f"{persona_id}-copy").strip().lower().replace(" ", "-")
    path = PERSONAS_DIR / f"{new_id}.yaml"
    if path.exists():
        raise HTTPException(status_code=409, detail=f"Persona '{new_id}' already exists")
    
    data["name"] = data.get("name", persona_id) + " (Copy)"
    _write_yaml(new_id, data)
    log.info("Duplicated persona '%s' -> '%s'", persona_id, new_id)
    return _persona_full(new_id, data)
