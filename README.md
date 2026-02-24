# 🦞 rayx-claw

A multi-agent orchestrator platform powered by Claude (Anthropic) with parallel MiniMax sub-agents, filesystem tools, persistent memory, and voice output.

## Overview

rayx-claw is a real-time conversational AI assistant that uses Claude as the main orchestrator brain. It can delegate tasks to multiple MiniMax sub-agents running in parallel, execute filesystem operations, persist memory across sessions, and speak responses aloud using a cloned voice (Voicebox TTS).

## Architecture

| File | Description |
|------|-------------|
| `server.py` | FastAPI server with WebSocket endpoints, serves static UI, exposes status/persona/token APIs |
| `orchestrator.py` | Main agent loop: streams Claude responses, executes tools, delegates to MiniMax sub-agents |
| `agents.py` | MiniMax API client (Anthropic-compatible) for sub-agent task delegation |
| `tools.py` | Tool implementations: filesystem ops, shell commands, memory (CRUD), TTS voice output |
| `storage.py` | Persistent storage: chat history (JSON), key-value memory (JSON) in `data/` directory |
| `auth.py` | OAuth token manager: auto-loads/refreshing tokens from Claude Code credentials |
| `config.py` | Pydantic Settings for environment variables (`.env` support) |
| `models.py` | Pydantic models for WebSocket messages and agent enum types |

## Prerequisites

- **Python 3.10+**
- [Claude Code](https://claude.com/code) with a Claude Max subscription (for OAuth tokens)
- MiniMax API key (for sub-agents)
- Optional: Voicebox server for TTS voice output

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/ElDiegolar/rayx-claw.git
cd rayx-claw

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings (see Configuration below)
```

## Running

```bash
# Start the server
python -m server

# Or with uvicorn directly
uvicorn server:app --host 127.0.0.1 --port 8080
```

The server runs on `http://127.0.0.1:8080` by default. Open the URL in your browser to access the web UI.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key (for OAuth, leave empty if using token manager) | (empty) |
| `MINIMAX_API_KEY` | MiniMax API key for sub-agents | (required) |
| `CLAUDE_MODEL` | Claude model for orchestrator | `claude-opus-4-6` |
| `MINIMAX_MODEL` | MiniMax model for sub-agents | `MiniMax-M2.5` |
| `WORKSPACE` | Filesystem root for agent operations | `E:\Sophia` |
| `PERSONA_NAME` | Assistant name (default persona) | `Sophia` |
| `HOST` | Server bind address | `127.0.0.1` |
| `PORT` | Server port | `8080` |

### OAuth Token Setup

rayx-claw automatically loads OAuth tokens from Claude Code's credentials file:

1. Ensure Claude Code is installed and signed in with a Claude Max subscription
2. The app reads `~/.claude/.credentials.json` for OAuth tokens
3. Tokens are auto-refreshed before expiry (10-minute buffer)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve web UI (`static/index.html`) |
| `/ws` | WebSocket | Real-time chat: send messages, receive streaming responses |
| `/api/status` | GET | Health check for all tools (Claude, MiniMax, Voicebox, etc.) |
| `/api/persona` | GET | Get current persona name |
| `/api/persona` | POST | Set persona name |
| `/api/token` | GET | OAuth token status (expiry time, remaining) |
| `/api/token/refresh` | POST | Force OAuth token refresh |
| `/static/*` | GET | Serve static files |

## WebSocket Protocol

Connect to `/ws` to chat. Messages are JSON:

**Client → Server:**
```json
{"type": "message", "content": "Hello!"}
{"type": "cancel"}
```

**Server → Client:**
```json
{"type": "chunk", "agent": "claude", "content": "Hello..."}
{"type": "tool_use", "agent": "claude", "tool_name": "read_file", "tool_input": {"path": "..."}}
{"type": "tool_result", "agent": "system", "content": "File contents...", "tool_name": "read_file"}
{"type": "status", "agent": "claude", "content": "Thinking..."}
{"type": "done", "agent": "system"}
{"type": "error", "agent": "system", "content": "..."}
{"type": "history", "agent": "system", "history": [...]}
```

## Tools Available

### Orchestrator (Claude)
- `read_file`, `write_file`, `list_directory`, `search_files`, `grep_files` — filesystem operations
- `run_command` — shell command execution (workspace-restricted)
- `delegate_to_minimax` — spawn parallel MiniMax sub-agents
- `speak` — text-to-speech via Voicebox (cloned voice)
- `save_memory`, `recall_memory`, `delete_memory` — persistent key-value memory

### Sub-agents (MiniMax)
- Filesystem tools only: `read_file`, `write_file`, `list_directory`, `search_files`, `grep_files`, `run_command`

## Tech Stack

- **Backend:** Python 3.10+, FastAPI, uvicorn
- **AI:** Anthropic Claude (OAuth), MiniMax (sub-agents)
- **Communication:** WebSockets (real-time streaming)
- **Storage:** JSON files (`data/history.json`, `data/memory.json`)
- **Voice:** Voicebox TTS (optional), Edge TTS fallback
- **UI:** Static HTML/JS (served from `static/`)

## License

MIT License — see LICENSE file for details.
