"""
Infinite Context Memory — Python Client SDK

HTTP client for the ICM server REST API with response parsing,
SSE stream handling, and session lifecycle management.
"""

import json
from typing import Iterator, Optional, Dict, Any, List, Generator

import requests


class IcmClientError(Exception):
    """Raised on API errors, unexpected responses, or connection failures."""


class IcmClient:
    """HTTP client for the Infinite Context Memory server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_response(data: dict) -> Dict[str, Any]:
        """Normalise an API JSON response into a predictable dict."""
        return {
            "response": data.get("response", ""),
            "session_id": data.get("session_id", ""),
            "turns_compressed": data.get("turns_compressed", 0),
            "memory_bytes": data.get("memory_bytes", 0),
        }

    @staticmethod
    def parse_stream_line(line: str) -> Optional[dict]:
        """Parse a single SSE line.  Returns None for comments / blanks."""
        line = line.strip()
        if not line or line.startswith(":"):
            return None
        if line.startswith("data: "):
            payload = line[6:]
            if payload == "[DONE]":
                return {"event": "done"}
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def parse_stream(lines: Iterator[str]) -> Iterator[dict]:
        """Parse an iterable of SSE lines into an event stream."""
        for line in lines:
            parsed = IcmClient.parse_stream_line(line)
            if parsed is not None:
                yield parsed

    # ------------------------------------------------------------------
    # API methods
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """Make an HTTP request, wrapping connection errors as IcmClientError."""
        url = f"{self.base_url}{path}"
        try:
            return self._session.request(method, url, **kwargs)
        except requests.exceptions.RequestException as e:
            raise IcmClientError(f"Request failed: {e}") from e

    def health(self) -> dict:
        resp = self._request("GET", "/health")
        if resp.status_code != 200:
            raise IcmClientError(f"Health check failed: HTTP {resp.status_code}")
        return resp.json()

    def chat(self, session_id: str, message: str) -> dict:
        resp = self._request(
            "POST", "/chat",
            json={"session_id": session_id, "message": message},
        )
        if resp.status_code == 404:
            raise IcmClientError("Session not found")
        if resp.status_code != 200:
            raise IcmClientError(
                f"Chat failed: HTTP {resp.status_code} {resp.text}"
            )
        return self.parse_response(resp.json())

    def list_sessions(self) -> List[dict]:
        resp = self._request("GET", "/sessions")
        if resp.status_code != 200:
            raise IcmClientError(
                f"List sessions failed: HTTP {resp.status_code}"
            )
        return resp.json().get("sessions", [])

    def get_session(self, session_id: str) -> dict:
        resp = self._request("GET", f"/sessions/{session_id}")
        if resp.status_code == 404:
            raise IcmClientError("Session not found")
        if resp.status_code != 200:
            raise IcmClientError(
                f"Get session failed: HTTP {resp.status_code}"
            )
        return resp.json()

    def delete_session(self, session_id: str) -> dict:
        resp = self._request("DELETE", f"/sessions/{session_id}")
        if resp.status_code != 200:
            raise IcmClientError(
                f"Delete session failed: HTTP {resp.status_code}"
            )
        return resp.json()

    def export_json(self, session_id: str) -> dict:
        """Export a session's conversation history as JSON."""
        resp = self._request("GET", f"/sessions/{session_id}/export/json")
        if resp.status_code == 404:
            raise IcmClientError("Session not found")
        if resp.status_code != 200:
            raise IcmClientError(
                f"Export JSON failed: HTTP {resp.status_code}"
            )
        return resp.json()

    def export_markdown(self, session_id: str) -> str:
        """Export a session's conversation history as Markdown text."""
        resp = self._request("GET", f"/sessions/{session_id}/export/markdown")
        if resp.status_code == 404:
            raise IcmClientError("Session not found")
        if resp.status_code != 200:
            raise IcmClientError(
                f"Export Markdown failed: HTTP {resp.status_code}"
            )
        return resp.text

    def chat_stream_ws(
        self, session_id: str, message: str,
        ws_url: Optional[str] = None,
    ) -> Generator[dict, None, None]:
        """Stream chat tokens via WebSocket.

        Yields dicts with keys:
          - {"token": str, "session_id": str} for each token
          - {"done": True, "session_id": str, "turns_compressed": int, "memory_bytes": int}
          - {"error": str} on failure

        Requires the ``websockets`` package.
        """
        try:
            import websockets.sync.client as ws_client
        except ImportError:
            raise IcmClientError(
                "WebSocket support requires `websockets`. "
                "Install: pip install websockets"
            )

        if ws_url is None:
            host = self.base_url.replace("http://", "").replace("https://", "")
            proto = "wss" if self.base_url.startswith("https") else "ws"
            ws_url = f"{proto}://{host}/chat/ws"

        try:
            with ws_client.connect(ws_url) as ws:
                ws.send(json.dumps({"session_id": session_id, "message": message}))
                while True:
                    raw = ws.recv()
                    if raw is None:
                        break
                    try:
                        data = json.loads(raw)
                        yield data
                        if data.get("done") or data.get("error"):
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            raise IcmClientError(f"WebSocket chat failed: {e}") from e

    def close(self):
        self._session.close()
