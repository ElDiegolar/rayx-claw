"""Persona loader — reads YAML persona definitions and builds system prompts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

PERSONAS_DIR = Path(__file__).parent / "personas"


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base recursively."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Persona:
    """Loaded persona with accessors and system prompt builder."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    # --- Identity ---
    @property
    def name(self) -> str:
        return self._data.get("name", "Claw")

    @property
    def tagline(self) -> str:
        return self._data.get("tagline", "AI assistant")

    @property
    def greeting(self) -> str:
        return self._data.get("greeting", f"Hello! I'm {self.name}. How can I help?")

    # --- Personality ---
    @property
    def tone(self) -> str:
        return self._data.get("personality", {}).get("tone", "professional")

    @property
    def style(self) -> str:
        return self._data.get("personality", {}).get("style", "concise and helpful")

    @property
    def traits(self) -> list[str]:
        return self._data.get("personality", {}).get("traits", [])

    # --- Role ---
    @property
    def role_description(self) -> str:
        return self._data.get("role", {}).get("description", "You are a capable AI assistant.")

    @property
    def instructions(self) -> list[str]:
        return self._data.get("role", {}).get("instructions", [])

    # --- Guardrails ---
    @property
    def guardrails_do(self) -> list[str]:
        return self._data.get("guardrails", {}).get("do", [])

    @property
    def guardrails_dont(self) -> list[str]:
        return self._data.get("guardrails", {}).get("dont", [])

    # --- Voice ---
    @property
    def voice_enabled(self) -> bool:
        return self._data.get("voice", {}).get("enabled", True)

    @property
    def voice_config(self) -> dict[str, Any]:
        return self._data.get("voice", {})

    @property
    def voicebox_host(self) -> str:
        return self._data.get("voice", {}).get("voicebox", {}).get("host", "127.0.0.1")

    @property
    def voicebox_port(self) -> int:
        return self._data.get("voice", {}).get("voicebox", {}).get("port", 17493)

    @property
    def voicebox_profile(self) -> str:
        return self._data.get("voice", {}).get("voicebox", {}).get("profile_id", "")

    @property
    def edge_tts_voice(self) -> str:
        return self._data.get("voice", {}).get("edge_tts", {}).get("voice", "en-US-AriaNeural")

    @property
    def edge_tts_rate(self) -> str:
        return self._data.get("voice", {}).get("edge_tts", {}).get("rate", "+0%")

    # --- UI ---
    @property
    def avatar_emoji(self) -> str:
        return self._data.get("ui", {}).get("avatar_emoji", "🤖")

    @property
    def primary_color(self) -> str:
        return self._data.get("ui", {}).get("primary_color", "#e94560")

    @property
    def theme(self) -> str:
        return self._data.get("ui", {}).get("theme", "dark")

    # --- Raw data ---
    @property
    def raw(self) -> dict[str, Any]:
        return self._data.copy()

    # --- System Prompt Builder ---
    def build_system_prompt(self, workspace: str) -> str:
        """Build the full system prompt from persona fields."""
        sections = []

        # Identity & role
        sections.append(f"You are {self.name}, {self.role_description.strip()}")

        # Personality
        if self.tone or self.style:
            sections.append(
                f"Communication style: {self.tone}. {self.style}."
            )

        if self.traits:
            traits_text = "\n".join(f"- {t}" for t in self.traits)
            sections.append(f"Personality traits:\n{traits_text}")

        # Tools
        sections.append("""Available tools:
- Filesystem tools: read_file, write_file, list_directory, search_files, grep_files
- run_command: Execute shell commands (start servers, run scripts, install packages, git, npm, pip, etc.). Commands run in workspace by default with configurable timeout.
- delegate_to_subagent: Send a task to a named sub-agent. Each agent_id gets its own conversation history, so you can have ongoing back-and-forth with multiple agents simultaneously.
- save_memory / recall_memory / delete_memory: Persistent memory across sessions.""")

        # Voice tool (only if enabled)
        if self.voice_enabled:
            sections.append(
                f"- speak: Speak text aloud using {self.name}'s voice. "
                "Use this to greet the user, narrate results, or whenever voice output adds value. "
                "Use fast=true for quick short responses."
            )

        # Instructions
        if self.instructions:
            instructions_text = "\n".join(f"- {i}" for i in self.instructions)
            sections.append(f"Instructions:\n{instructions_text}")

        # Guardrails
        guardrails_parts = []
        if self.guardrails_do:
            guardrails_parts.append("Always:\n" + "\n".join(f"- {g}" for g in self.guardrails_do))
        if self.guardrails_dont:
            guardrails_parts.append("Never:\n" + "\n".join(f"- {g}" for g in self.guardrails_dont))
        if guardrails_parts:
            sections.append("Guardrails:\n" + "\n".join(guardrails_parts))

        # Workspace
        sections.append(f"Workspace root: {workspace}")

        return "\n\n".join(sections)

    def to_api_dict(self) -> dict[str, Any]:
        """Return persona info for the API."""
        return {
            "name": self.name,
            "tagline": self.tagline,
            "greeting": self.greeting,
            "avatar_emoji": self.avatar_emoji,
            "primary_color": self.primary_color,
            "theme": self.theme,
            "voice_enabled": self.voice_enabled,
        }


def load_persona(name: str = "default") -> Persona:
    """Load a persona by name from the personas/ directory.

    Always loads 'default.yaml' first as the base, then merges
    the requested persona on top (if different from default).
    """
    # Load default as base
    default_path = PERSONAS_DIR / "default.yaml"
    if not default_path.exists():
        log.warning("Default persona not found at %s, using built-in defaults", default_path)
        return Persona({})

    with open(default_path, "r", encoding="utf-8") as f:
        base_data = yaml.safe_load(f) or {}

    if name == "default":
        log.info("Loaded persona: default (%s)", base_data.get("name", "unknown"))
        return Persona(base_data)

    # Load and merge custom persona
    custom_path = PERSONAS_DIR / f"{name}.yaml"
    if not custom_path.exists():
        log.warning("Persona '%s' not found, falling back to default", name)
        return Persona(base_data)

    with open(custom_path, "r", encoding="utf-8") as f:
        custom_data = yaml.safe_load(f) or {}

    merged = _deep_merge(base_data, custom_data)
    log.info("Loaded persona: %s (%s)", name, merged.get("name", "unknown"))
    return Persona(merged)
