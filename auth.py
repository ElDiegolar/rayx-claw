"""OAuth token management: auto-sync from Claude Code credentials + refresh."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

CREDENTIALS_FILE = Path.home() / ".claude" / ".credentials.json"
OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
# Refresh 10 minutes before expiry
REFRESH_BUFFER_MS = 10 * 60 * 1000


class TokenManager:
    """Manages OAuth access tokens with auto-refresh."""

    def __init__(self) -> None:
        self.access_token: str = ""
        self.refresh_token: str = ""
        self.expires_at: int = 0  # epoch ms
        self._load_credentials()

    def _load_credentials(self) -> None:
        """Load tokens from Claude Code's credentials file."""
        if not CREDENTIALS_FILE.exists():
            log.warning("Credentials file not found: %s", CREDENTIALS_FILE)
            return
        try:
            data = json.loads(CREDENTIALS_FILE.read_text("utf-8"))
            oauth = data.get("claudeAiOauth", {})
            self.access_token = oauth.get("accessToken", "")
            self.refresh_token = oauth.get("refreshToken", "")
            self.expires_at = oauth.get("expiresAt", 0)
            log.info(
                "Loaded OAuth token (expires in %.1f hours)",
                max(0, (self.expires_at - _now_ms()) / 3_600_000),
            )
        except Exception:
            log.exception("Failed to load credentials")

    def _save_credentials(self) -> None:
        """Write updated tokens back to Claude Code's credentials file."""
        if not CREDENTIALS_FILE.exists():
            return
        try:
            data = json.loads(CREDENTIALS_FILE.read_text("utf-8"))
            data["claudeAiOauth"]["accessToken"] = self.access_token
            data["claudeAiOauth"]["refreshToken"] = self.refresh_token
            data["claudeAiOauth"]["expiresAt"] = self.expires_at
            CREDENTIALS_FILE.write_text(json.dumps(data, indent=4), "utf-8")
            log.info("Saved refreshed tokens to %s", CREDENTIALS_FILE)
        except Exception:
            log.exception("Failed to save credentials")

    def is_expired(self) -> bool:
        return _now_ms() >= (self.expires_at - REFRESH_BUFFER_MS)

    def time_remaining_ms(self) -> int:
        return max(0, self.expires_at - _now_ms())

    def get_token(self) -> str:
        """Return current access token, refreshing first if needed."""
        if self.is_expired() and self.refresh_token:
            self._refresh_sync()
        return self.access_token

    async def get_token_async(self) -> str:
        """Async version — refreshes if needed."""
        if self.is_expired() and self.refresh_token:
            await self._refresh_async()
        return self.access_token

    def _refresh_sync(self) -> None:
        """Synchronous token refresh."""
        log.info("Refreshing OAuth token (sync)...")
        try:
            r = httpx.post(
                OAUTH_TOKEN_URL,
                json={
                    "grant_type": "refresh_token",
                    "client_id": OAUTH_CLIENT_ID,
                    "refresh_token": self.refresh_token,
                },
                timeout=15,
            )
            r.raise_for_status()
            self._apply_refresh(r.json())
        except Exception:
            log.exception("Token refresh failed (sync)")
            # Reload from file in case Claude Code refreshed it
            self._load_credentials()

    async def _refresh_async(self) -> None:
        """Async token refresh."""
        log.info("Refreshing OAuth token (async)...")
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(
                    OAUTH_TOKEN_URL,
                    json={
                        "grant_type": "refresh_token",
                        "client_id": OAUTH_CLIENT_ID,
                        "refresh_token": self.refresh_token,
                    },
                )
                r.raise_for_status()
                self._apply_refresh(r.json())
        except Exception:
            log.exception("Token refresh failed (async)")
            self._load_credentials()

    def _apply_refresh(self, data: dict) -> None:
        """Apply refresh response and persist."""
        self.access_token = data["access_token"]
        self.refresh_token = data["refresh_token"]
        expires_in = data.get("expires_in", 86400)
        # Store with 5-min safety buffer like Claude Code does
        self.expires_at = _now_ms() + (expires_in * 1000) - (5 * 60 * 1000)
        log.info(
            "Token refreshed, expires in %.1f hours",
            (self.expires_at - _now_ms()) / 3_600_000,
        )
        self._save_credentials()

    def get_info(self) -> dict:
        """Return token status for the UI."""
        remaining = self.time_remaining_ms()
        return {
            "expires_at": self.expires_at,
            "remaining_ms": remaining,
            "remaining_hours": round(remaining / 3_600_000, 2),
            "expired": self.is_expired(),
            "has_refresh": bool(self.refresh_token),
        }


def _now_ms() -> int:
    return int(time.time() * 1000)


# Singleton
token_manager = TokenManager()
