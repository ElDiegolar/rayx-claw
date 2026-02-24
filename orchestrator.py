from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from agents import call_minimax, call_minimax_stream, minimax_client
from auth import token_manager
from config import Settings
from models import Agent, WSMessage
from storage import HistoryStore, MemoryStore
from tools import TOOL_DEFINITIONS, execute_tool, memory_store

import anthropic
import httpx

log = logging.getLogger(__name__)
settings = Settings()


# OAuth auth for Claude — fetches current token dynamically
class _OAuthAuth(httpx.Auth):
    def auth_flow(self, request):
        request.headers.pop("x-api-key", None)
        request.headers["Authorization"] = f"Bearer {token_manager.get_token()}"
        request.headers["anthropic-beta"] = "oauth-2025-04-20"
        yield request


_claude = anthropic.AsyncAnthropic(
    api_key="placeholder",
    http_client=httpx.AsyncClient(auth=_OAuthAuth()),
)

# Type alias for the send callback
Send = Callable[[WSMessage], Awaitable[None]]

SYSTEM_PROMPT_BASE = f"""You are Sophia, a capable AI assistant with access to tools. You can spin up multiple MiniMax sub-agents to work on tasks in parallel, and you can speak aloud.

Available tools:
- Filesystem tools: read_file, write_file, list_directory, search_files, grep_files
- run_command: Execute shell commands (start servers, run scripts, install packages, git, npm, pip, etc.). Commands run in workspace by default with configurable timeout.
- delegate_to_minimax: Send a task to a named MiniMax sub-agent. Each agent_id gets its own conversation history, so you can have ongoing back-and-forth with multiple agents simultaneously.
- speak: Speak text aloud using your voice. Use this to greet the user, narrate results, or whenever voice output adds value. Use fast=true for quick short responses.
- save_memory / recall_memory / delete_memory: Persistent memory across sessions.

Multi-agent strategy:
- Give each agent a descriptive agent_id (e.g. "researcher", "coder", "reviewer", "writer")
- You can call delegate_to_minimax multiple times in a single turn — they run in PARALLEL
- Each named agent remembers its conversation, so you can follow up: "coder" remembers what it wrote
- Sub-agents have their own tools: read_file, write_file, list_directory, search_files, grep_files, run_command — they can explore the filesystem and execute commands autonomously
- Use different system prompts to specialize agents for different roles
- Combine results from multiple agents to produce better output

You are the lead orchestrator. You decide what to do, what to delegate, and how to combine results. Be conversational and helpful.

Workspace root: {settings.workspace}"""

# Add delegate_to_minimax to the tool list
ALL_TOOLS = TOOL_DEFINITIONS + [
    {
        "name": "delegate_to_minimax",
        "description": (
            "Delegate a task to a named MiniMax sub-agent. Each agent_id maintains its own "
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
    },
]


# Tools available to MiniMax sub-agents (no delegation, no speak, no memory)
SUBAGENT_TOOL_NAMES = {"read_file", "write_file", "list_directory", "search_files", "grep_files", "run_command"}
SUBAGENT_TOOLS = [t for t in TOOL_DEFINITIONS if t["name"] in SUBAGENT_TOOL_NAMES]


class CancelledError(Exception):
    """Raised when the user cancels the current task."""
    pass


class Orchestrator:
    """Conversational agent: Claude + filesystem tools + MiniMax delegation."""

    def __init__(self) -> None:
        self.history = HistoryStore()
        # Load previous conversation context
        self.messages: list[dict] = self.history.get_api_messages()
        if self.messages:
            log.info("Loaded %d messages from history", len(self.messages))
        # Named MiniMax agent conversations: agent_id -> list of messages
        self.minimax_agents: dict[str, list[dict]] = {}
        # Cancellation
        self._cancel_event = asyncio.Event()
        # Repair any dangling tool_use from previous interrupted session
        self._repair_messages()

    def cancel(self) -> None:
        """Signal the current task to stop."""
        self._cancel_event.set()

    def _check_cancelled(self) -> None:
        """Raise if cancelled."""
        if self._cancel_event.is_set():
            raise CancelledError("Task cancelled by user")

    def _build_system_prompt(self) -> str:
        """Build system prompt with memory context injected."""
        mem_context = memory_store.get_context()
        if mem_context:
            return SYSTEM_PROMPT_BASE + "\n\n" + mem_context
        return SYSTEM_PROMPT_BASE

    def _repair_messages(self) -> None:
        """Fix dangling tool_use blocks that have no tool_result.

        Scans the full message list. Every assistant message containing
        tool_use blocks must be immediately followed by a user message
        whose content includes a tool_result for each tool_use id.
        Missing results get stub entries injected.
        """
        if not self.messages:
            return

        repaired = []
        i = 0
        while i < len(self.messages):
            msg = self.messages[i]
            repaired.append(msg)

            # Check if this is an assistant message with tool_use blocks
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
                        # Check if next message has matching tool_results
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
                                # Append stubs to existing tool_result message
                                next_msg["content"].extend(stubs)
                            else:
                                # Insert a new user message with stub results
                                repaired.append({"role": "user", "content": stubs})
            i += 1

        self.messages = repaired

    def get_ui_history(self) -> list[dict]:
        """Return UI history for replay on connect."""
        return self.history.get_ui_history()

    async def run(self, user_message: str, send: Send) -> None:
        """Handle a user message through the tool-use loop."""
        self._cancel_event.clear()
        self.messages.append({"role": "user", "content": user_message})

        # Collect UI messages for history
        ui_messages: list[dict] = []
        assistant_text_parts: list[str] = []

        async def send_and_record(msg: WSMessage):
            ui_messages.append(msg.model_dump())
            try:
                await send(msg)
            except Exception:
                pass  # WebSocket already closed

        async def _safe_send(msg: WSMessage):
            try:
                await send(msg)
            except Exception:
                pass  # WebSocket already closed

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
            # Always repair dangling tool_use blocks before next request
            self._repair_messages()
            await _safe_send(WSMessage(type="done", agent=Agent.SYSTEM))
            # Save exchange to history
            assistant_text = "\n".join(assistant_text_parts)
            self.history.add_exchange(user_message, assistant_text, ui_messages)

    async def _agent_loop(self, send: Send, text_parts: list[str]) -> None:
        """Claude's main loop: respond, use tools, delegate, repeat."""
        max_rounds = 20

        for round_num in range(max_rounds):
            self._check_cancelled()

            if round_num > 0:
                await send(WSMessage(
                    type="status", agent=Agent.CLAUDE,
                    content=f"{settings.persona_name} thinking... (round {round_num + 1})",
                ))

            # Stream Claude's response for real-time feedback
            response = await self._stream_claude_response(send, text_parts)

            # Collect tool uses from response
            tool_uses = [b for b in response.content if b.type == "tool_use"]

            # If no tool calls, we're done
            if response.stop_reason == "end_turn" or not tool_uses:
                self.messages.append({"role": "assistant", "content": response.content})
                break

            # Execute tools
            self.messages.append({"role": "assistant", "content": response.content})

            # Partition into MiniMax delegations (parallel) and other tools (sequential)
            minimax_calls = []
            other_calls = []
            for tool_use in tool_uses:
                if tool_use.name == "delegate_to_minimax":
                    minimax_calls.append(tool_use)
                else:
                    other_calls.append(tool_use)

            tool_results = []

            # Execute non-MiniMax tools sequentially first
            self._check_cancelled()
            for i, tool_use in enumerate(other_calls):
                name = tool_use.name
                inp = dict(tool_use.input)

                await send(WSMessage(
                    type="status", agent=Agent.CLAUDE,
                    content=f"{settings.persona_name} running {name}..." + (f" ({i+1}/{len(other_calls)})" if len(other_calls) > 1 else ""),
                ))
                await send(WSMessage(
                    type="tool_use", agent=Agent.CLAUDE,
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

            # Execute MiniMax delegations in parallel
            if minimax_calls:
                agent_ids = [dict(tc.input).get("agent_id", "default") for tc in minimax_calls]
                await send(WSMessage(
                    type="status", agent=Agent.MINIMAX,
                    content=f"Launching {len(minimax_calls)} agent(s): {', '.join(agent_ids)}",
                ))

                async def _run_minimax(tool_use):
                    inp = dict(tool_use.input)
                    agent_id = inp.get("agent_id", "default")

                    await send(WSMessage(
                        type="tool_use", agent=Agent.CLAUDE,
                        tool_name="delegate_to_minimax",
                        tool_input=inp,
                    ))

                    result_text = await self._delegate_minimax(
                        agent_id, inp["prompt"], inp.get("system", ""), send
                    )
                    log.info("MiniMax[%s] -> %d chars", agent_id, len(result_text))

                    display = result_text[:2000] + "..." if len(result_text) > 2000 else result_text
                    await send(WSMessage(
                        type="tool_result", agent=Agent.SYSTEM,
                        content=display, tool_name=f"minimax:{agent_id}",
                    ))

                    return {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": result_text,
                    }

                minimax_results = await asyncio.gather(
                    *[_run_minimax(tc) for tc in minimax_calls]
                )
                tool_results.extend(minimax_results)

            self.messages.append({"role": "user", "content": tool_results})

    async def _stream_claude_response(self, send: Send, text_parts: list[str]):
        """Stream Claude's response, sending text chunks to UI in real time."""
        collected_text = []

        async with _claude.messages.stream(
            model=settings.claude_model,
            max_tokens=4096,
            system=self._build_system_prompt(),
            messages=self.messages,
            tools=ALL_TOOLS,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text") and event.delta.text:
                        collected_text.append(event.delta.text)
                        await send(WSMessage(
                            type="chunk", agent=Agent.CLAUDE,
                            content=event.delta.text,
                        ))

            response = await stream.get_final_message()

        # Record full text for history
        full_text = "".join(collected_text)
        if full_text.strip():
            text_parts.append(full_text)

        return response

    async def _delegate_minimax(
        self, agent_id: str, prompt: str, system: str, send: Send
    ) -> str:
        """Delegate a task to a named MiniMax agent with tool use loop."""
        # Initialize agent conversation if new
        if agent_id not in self.minimax_agents:
            self.minimax_agents[agent_id] = []
            log.info("Spawning new MiniMax agent: %s", agent_id)

        agent_messages = self.minimax_agents[agent_id]
        agent_messages.append({"role": "user", "content": prompt})

        # Only use system prompt on first message (or if explicitly provided)
        effective_system = system if (system or len(agent_messages) == 1) else ""

        max_tool_rounds = 15
        full_text = ""

        for round_num in range(max_tool_rounds):
            self._check_cancelled()

            if round_num > 0:
                await send(WSMessage(
                    type="status", agent=Agent.MINIMAX,
                    content=f"MiniMax [{agent_id}] continuing... (round {round_num + 1})",
                ))
            else:
                await send(WSMessage(
                    type="status", agent=Agent.MINIMAX,
                    content=f"MiniMax [{agent_id}] working...",
                ))

            # Stream response with tools
            round_text = ""
            async with minimax_client.messages.stream(
                model=settings.minimax_model,
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
                                type="chunk", agent=Agent.MINIMAX,
                                content=event.delta.text,
                                tool_name=agent_id,
                            ))

                response = await stream.get_final_message()

            full_text += round_text

            # Save assistant response to conversation
            agent_messages.append({"role": "assistant", "content": response.content})

            # Check for tool uses
            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if not tool_uses or response.stop_reason == "end_turn":
                break

            # Execute tools
            tool_results = []
            for tu in tool_uses:
                inp = dict(tu.input)
                await send(WSMessage(
                    type="tool_use", agent=Agent.MINIMAX,
                    tool_name=tu.name, tool_input=inp,
                ))

                result_text = execute_tool(tu.name, inp)
                log.info("MiniMax[%s] tool %s -> %d chars", agent_id, tu.name, len(result_text))

                display = result_text[:2000] + "..." if len(result_text) > 2000 else result_text
                await send(WSMessage(
                    type="tool_result", agent=Agent.SYSTEM,
                    content=display, tool_name=f"{agent_id}:{tu.name}",
                ))

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result_text,
                })

            agent_messages.append({"role": "user", "content": tool_results})

        log.info(
            "MiniMax[%s] conversation: %d messages",
            agent_id, len(agent_messages),
        )

        return full_text
