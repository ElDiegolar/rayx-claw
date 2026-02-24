from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from agents import minimax_client
from auth import token_manager
from config import Settings
from models import Agent, WSMessage
from persona import load_persona
from storage import HistoryStore
from tools import TOOL_DEFINITIONS, execute_tool, memory_store

import anthropic
import httpx

log = logging.getLogger(__name__)
settings = Settings()


# ---------------------------------------------------------------------------
# LLM Clients
# ---------------------------------------------------------------------------

# OAuth auth for Claude — fetches current token dynamically
class _OAuthAuth(httpx.Auth):
    def auth_flow(self, request):
        request.headers.pop("x-api-key", None)
        request.headers["Authorization"] = f"Bearer {token_manager.get_token()}"
        request.headers["anthropic-beta"] = "oauth-2025-04-20"
        yield request


claude_client = anthropic.AsyncAnthropic(
    api_key="placeholder",
    http_client=httpx.AsyncClient(auth=_OAuthAuth()),
)

# Provider config lookup: client, model, max_tokens for lead role
PROVIDERS = {
    "claude": {
        "client": claude_client,
        "model": lambda: settings.claude_model,
        "max_tokens": 8096,
        "label": "Claude",
        "agent": Agent.CLAUDE,
    },
    "minimax": {
        "client": minimax_client,
        "model": lambda: settings.minimax_model,
        "max_tokens": 16384,
        "label": "MiniMax",
        "agent": Agent.MINIMAX,
    },
}


# Type alias for the send callback
Send = Callable[[WSMessage], Awaitable[None]]


# ---------------------------------------------------------------------------
# Tool definitions (dynamic based on provider)
# ---------------------------------------------------------------------------

# Tools available to sub-agents (no delegation, no speak, no memory)
SUBAGENT_TOOL_NAMES = {"read_file", "write_file", "list_directory", "search_files", "grep_files", "run_command"}
SUBAGENT_TOOLS = [t for t in TOOL_DEFINITIONS if t["name"] in SUBAGENT_TOOL_NAMES]

DELEGATE_TOOL_NAME = "delegate_to_subagent"


def _build_delegation_tool(sub_label: str) -> dict:
    """Build the delegation tool definition targeting the current sub-agent."""
    return {
        "name": DELEGATE_TOOL_NAME,
        "description": (
            f"Delegate a task to a named {sub_label} sub-agent. Each agent_id maintains its own "
            "conversation history, so you can have ongoing multi-turn conversations with "
            "multiple agents. Call this tool multiple times in one turn to run agents in "
            "PARALLEL. Use descriptive agent_ids like 'researcher', 'coder', 'reviewer'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": (
                        "Name for this agent instance (e.g. 'coder', 'researcher', 'reviewer'). "
                        "Re-use the same id to continue a conversation with that agent."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": "The task prompt or follow-up message for the sub-agent",
                },
                "system": {
                    "type": "string",
                    "description": "Optional system prompt to set the sub-agent's role. Only used on first message to this agent_id.",
                    "default": "",
                },
            },
            "required": ["agent_id", "prompt"],
        },
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class CancelledError(Exception):
    """Raised when the user cancels the current task."""
    pass


class Orchestrator:
    """Provider-agnostic conversational agent with tool use and sub-agent delegation."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider or settings.orchestrator_provider
        if self.provider not in PROVIDERS:
            log.warning("Unknown provider '%s', falling back to claude", self.provider)
            self.provider = "claude"

        self._lead = PROVIDERS[self.provider]
        self._sub_key = "minimax" if self.provider == "claude" else "claude"
        self._sub = PROVIDERS[self._sub_key]

        log.info(
            "Orchestrator init: lead=%s (%s), sub=%s (%s)",
            self.provider, self._lead["model"](),
            self._sub_key, self._sub["model"](),
        )

        # Load the active persona
        self._persona = load_persona(settings.persona_name)

        # Build tool list with the delegation tool targeting the sub-agent
        self._all_tools = TOOL_DEFINITIONS + [_build_delegation_tool(self._sub["label"])]

        self.history = HistoryStore()
        self.messages: list[dict] = self.history.get_api_messages()
        if self.messages:
            log.info("Loaded %d messages from history", len(self.messages))

        # Named sub-agent conversations: agent_id -> list of messages
        self.subagent_convos: dict[str, list[dict]] = {}

        # Cancellation
        self._cancel_event = asyncio.Event()
        self._repair_messages()

    def cancel(self) -> None:
        self._cancel_event.set()

    def _check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise CancelledError("Task cancelled by user")

    def _build_system_prompt(self) -> str:
        # Use persona's system prompt builder
        system = self._persona.build_system_prompt(settings.workspace)
        
        # Add memory context
        mem_context = memory_store.get_context()
        if mem_context:
            system += "\n\n" + mem_context
        return system

    def _repair_messages(self) -> None:
        """Fix dangling tool_use blocks that have no tool_result."""
        if not self.messages:
            return

        repaired = []
        i = 0
        while i < len(self.messages):
            msg = self.messages[i]
            repaired.append(msg)

            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, list):
                    tool_use_ids = set()
                    for b in content:
                        btype = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
                        bid = b.get("id") if isinstance(b, dict) else getattr(b, "id", None)
                        if btype == "tool_use" and bid:
                            tool_use_ids.add(bid)

                    if tool_use_ids:
                        next_msg = self.messages[i + 1] if i + 1 < len(self.messages) else None
                        covered_ids = set()
                        if next_msg and next_msg.get("role") == "user":
                            next_content = next_msg.get("content")
                            if isinstance(next_content, list):
                                for r in next_content:
                                    if isinstance(r, dict) and r.get("type") == "tool_result":
                                        covered_ids.add(r.get("tool_use_id"))

                        missing = tool_use_ids - covered_ids
                        if missing:
                            log.warning(
                                "Repairing %d dangling tool_use block(s) at message %d",
                                len(missing), i,
                            )
                            stubs = [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tid,
                                    "content": "[interrupted — tool was not executed]",
                                }
                                for tid in missing
                            ]
                            if next_msg and next_msg.get("role") == "user" and isinstance(next_msg.get("content"), list):
                                next_msg["content"].extend(stubs)
                            else:
                                repaired.append({"role": "user", "content": stubs})
            i += 1

        self.messages = repaired

    def get_ui_history(self) -> list[dict]:
        return self.history.get_ui_history()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, user_message: str, send: Send) -> None:
        self._cancel_event.clear()
        self.messages.append({"role": "user", "content": user_message})

        ui_messages: list[dict] = []
        assistant_text_parts: list[str] = []

        async def send_and_record(msg: WSMessage):
            ui_messages.append(msg.model_dump())
            try:
                await send(msg)
            except Exception:
                pass

        async def _safe_send(msg: WSMessage):
            try:
                await send(msg)
            except Exception:
                pass

        try:
            await self._agent_loop(send_and_record, assistant_text_parts)
        except (CancelledError, asyncio.CancelledError):
            log.info("Task cancelled by user")
            await _safe_send(WSMessage(
                type="status", agent=Agent.SYSTEM, content="Cancelled.",
            ))
        except Exception as exc:
            log.exception("Orchestration error")
            await _safe_send(WSMessage(type="error", agent=Agent.SYSTEM, content=str(exc)))
        finally:
            self._repair_messages()
            await _safe_send(WSMessage(type="done", agent=Agent.SYSTEM))
            assistant_text = "\n".join(assistant_text_parts)
            self.history.add_exchange(user_message, assistant_text, ui_messages)

    # ------------------------------------------------------------------
    # Agent loop (provider-agnostic)
    # ------------------------------------------------------------------

    async def _agent_loop(self, send: Send, text_parts: list[str]) -> None:
        """Main agent loop: stream lead response, execute tools, delegate, repeat."""
        lead_agent = self._lead["agent"]
        sub_agent = self._sub["agent"]
        max_rounds = 20
        persona_name = self._persona.name

        for round_num in range(max_rounds):
            self._check_cancelled()

            if round_num > 0:
                await send(WSMessage(
                    type="status", agent=lead_agent,
                    content=f"{persona_name} thinking... (round {round_num + 1})",
                ))

            # Stream lead model's response
            response = await self._stream_lead_response(send, text_parts)

            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if response.stop_reason == "end_turn" or not tool_uses:
                self.messages.append({"role": "assistant", "content": response.content})
                break

            self.messages.append({"role": "assistant", "content": response.content})

            # Partition: delegation calls (parallel) vs other tools (sequential)
            delegation_calls = []
            other_calls = []
            for tool_use in tool_uses:
                if tool_use.name == DELEGATE_TOOL_NAME:
                    delegation_calls.append(tool_use)
                else:
                    other_calls.append(tool_use)

            tool_results = []

            # Execute non-delegation tools sequentially
            self._check_cancelled()
            for i, tool_use in enumerate(other_calls):
                name = tool_use.name
                inp = dict(tool_use.input)

                await send(WSMessage(
                    type="status", agent=lead_agent,
                    content=f"{persona_name} running {name}..." + (f" ({i+1}/{len(other_calls)})" if len(other_calls) > 1 else ""),
                ))
                await send(WSMessage(
                    type="tool_use", agent=lead_agent,
                    tool_name=name, tool_input=inp,
                ))

                result_text = execute_tool(name, inp)
                log.info("Tool %s -> %d chars", name, len(result_text))

                display = result_text[:2000] + "..." if len(result_text) > 2000 else result_text
                await send(WSMessage(
                    type="tool_result", agent=Agent.SYSTEM,
                    content=display, tool_name=name,
                ))

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result_text,
                })

            # Execute delegations in parallel
            if delegation_calls:
                agent_ids = [dict(tc.input).get("agent_id", "default") for tc in delegation_calls]
                await send(WSMessage(
                    type="status", agent=sub_agent,
                    content=f"Launching {len(delegation_calls)} {self._sub['label']} agent(s): {', '.join(agent_ids)}",
                ))

                async def _run_delegation(tool_use):
                    inp = dict(tool_use.input)
                    agent_id = inp.get("agent_id", "default")

                    await send(WSMessage(
                        type="tool_use", agent=lead_agent,
                        tool_name=DELEGATE_TOOL_NAME,
                        tool_input=inp,
                    ))

                    result_text = await self._delegate_to_subagent(
                        agent_id, inp["prompt"], inp.get("system", ""), send
                    )
                    log.info("%s[%s] -> %d chars", self._sub["label"], agent_id, len(result_text))

                    display = result_text[:2000] + "..." if len(result_text) > 2000 else result_text
                    await send(WSMessage(
                        type="tool_result", agent=Agent.SYSTEM,
                        content=display, tool_name=f"{self._sub_key}:{agent_id}",
                    ))

                    return {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": result_text,
                    }

                delegation_results = await asyncio.gather(
                    *[_run_delegation(tc) for tc in delegation_calls]
                )
                tool_results.extend(delegation_results)

            self.messages.append({"role": "user", "content": tool_results})

    # ------------------------------------------------------------------
    # Lead model streaming
    # ------------------------------------------------------------------

    async def _stream_lead_response(self, send: Send, text_parts: list[str]):
        """Stream the lead model's response, sending text chunks to UI."""
        client = self._lead["client"]
        model = self._lead["model"]()
        max_tokens = self._lead["max_tokens"]
        lead_agent = self._lead["agent"]

        collected_text = []

        async with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=self._build_system_prompt(),
            messages=self.messages,
            tools=self._all_tools,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text") and event.delta.text:
                        collected_text.append(event.delta.text)
                        await send(WSMessage(
                            type="chunk", agent=lead_agent,
                            content=event.delta.text,
                        ))

            response = await stream.get_final_message()

        full_text = "".join(collected_text)
        if full_text.strip():
            text_parts.append(full_text)

        return response

    # ------------------------------------------------------------------
    # Sub-agent delegation
    # ------------------------------------------------------------------

    async def _delegate_to_subagent(
        self, agent_id: str, prompt: str, system: str, send: Send
    ) -> str:
        """Delegate a task to a named sub-agent with tool use loop."""
        sub_client = self._sub["client"]
        sub_model = self._sub["model"]()
        sub_label = self._sub["label"]
        sub_agent = self._sub["agent"]

        if agent_id not in self.subagent_convos:
            self.subagent_convos[agent_id] = []
            log.info("Spawning new %s agent: %s", sub_label, agent_id)

        agent_messages = self.subagent_convos[agent_id]
        agent_messages.append({"role": "user", "content": prompt})

        effective_system = system if (system or len(agent_messages) == 1) else ""

        max_tool_rounds = 15
        full_text = ""

        for round_num in range(max_tool_rounds):
            self._check_cancelled()

            if round_num > 0:
                await send(WSMessage(
                    type="status", agent=sub_agent,
                    content=f"{sub_label} [{agent_id}] continuing... (round {round_num + 1})",
                ))
            else:
                await send(WSMessage(
                    type="status", agent=sub_agent,
                    content=f"{sub_label} [{agent_id}] working...",
                ))

            round_text = ""
            async with sub_client.messages.stream(
                model=sub_model,
                max_tokens=16384,
                system=effective_system or anthropic.NOT_GIVEN,
                messages=agent_messages,
                tools=SUBAGENT_TOOLS,
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text") and event.delta.text:
                            round_text += event.delta.text
                            await send(WSMessage(
                                type="chunk", agent=sub_agent,
                                content=event.delta.text,
                                tool_name=agent_id,
                            ))

                response = await stream.get_final_message()

            full_text += round_text
            agent_messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break

            # Execute tools
            tool_results = []
            for tool_use in tool_uses:
                name = tool_use.name
                inp = dict(tool_use.input)

                await send(WSMessage(
                    type="status", agent=sub_agent,
                    content=f"{sub_label} [{agent_id}] running {name}...",
                ))
                await send(WSMessage(
                    type="tool_use", agent=sub_agent,
                    tool_name=name, tool_input=inp,
                ))

                result_text = execute_tool(name, inp)
                log.info("Sub-agent %s[%s] tool %s -> %d chars", sub_label, agent_id, name, len(result_text))

                display = result_text[:2000] + "..." if len(result_text) > 2000 else result_text
                await send(WSMessage(
                    type="tool_result", agent=Agent.SYSTEM,
                    content=display, tool_name=name,
                ))

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result_text,
                })

            agent_messages.append({"role": "user", "content": tool_results})

        return full_text
