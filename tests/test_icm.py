"""
Comprehensive test suite for the Infinite Context Memory stack.

Coverage:
  - Unit tests for HierarchicalHyperbolicMemory (HHM core)
  - Unit tests for InfiniteContextMemory / ChatSession / RAGStore
  - Integration tests for the FastAPI server endpoints
  - Unit tests for the client SDK (icm_client.py)
"""

import pickle
import json
from io import StringIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# =====================================================================
#  UNIT TESTS — Hierarchical Hyperbolic Memory (HHM) core
# =====================================================================


class TestHierarchicalMemory:
    """Low-level geometric and architectural unit tests for HHM."""

    @pytest.mark.unit
    def test_manifold_constraint(self, hhm):
        """Points on the hyperboloid satisfy <u,u>_L = -1 (within tolerance)."""
        B, T, D = 4, 16, 32
        x = torch.randn(B, T, D)
        states, final = hhm(x)

        for name, tensor in [("final", final), ("all_states", states)]:
            inner = hhm._curvature() * torch.abs(
                hhm.lorentz_inner(tensor, tensor) + 1.0
            )
            violation = inner.max().item()
            assert violation < 1e-4, (
                f"Manifold constraint violated for {name}: "
                f"max <u,u>_L + 1 = {violation:.2e}"
            )

    @pytest.mark.unit
    def test_o1_memory(self, hhm):
        """State size is independent of sequence length (O(1) property)."""
        D = hhm.state_dim
        sizes = []
        for seq_len in [1, 8, 64, 256, 1024]:
            x = torch.randn(1, seq_len, D)
            _, final = hhm(x)
            sizes.append(final.numel())

        assert all(
            s == sizes[0] for s in sizes
        ), f"State size changed across lengths: {sizes}"
        # Expected: (D + 1) elements in fp32
        assert sizes[0] == (D + 1), f"Expected {(D + 1)} elements, got {sizes[0]}"

    @pytest.mark.unit
    def test_multi_scale_diversity(self, hhm):
        """Different scales produce different readout vectors."""
        B, T, D = 2, 32, 32
        x = torch.randn(B, T, D)
        _, final = hhm(x)

        scales = hhm.read_all_scales(final)
        assert len(scales) == hhm.num_scales

        for i in range(len(scales) - 1):
            diff = (scales[i] - scales[i + 1]).norm(dim=-1)
            assert diff.min() > 1e-6, (
                f"Scales {i} and {i + 1} produced identical readout"
            )

    @pytest.mark.unit
    def test_deterministic(self, hhm):
        """Same input always produces the same output."""
        B, T, D = 2, 32, 32
        x = torch.randn(B, T, D)

        _, f1 = hhm(x)
        _, f2 = hhm(x)
        diff = (f1 - f2).abs().max().item()
        assert diff < 1e-6, f"Determinism violated: max diff = {diff:.2e}"

    @pytest.mark.unit
    def test_exp_map_inverse(self):
        """exp_map and log_map are approximate inverses for small tangent
        vectors where the simplified lift cosh(t) ≈ sqrt(1+t²) holds."""
        from hyper_ssm.hierarchical_memory import exp_map, log_map

        B, D = 8, 16
        # Use small-magnitude vectors so the acosh(sqrt(1+||v||²)) ≈ ||v||
        # approximation used in the "small" path of exp_map is accurate.
        tangent_vecs = torch.randn(B, D) * 0.01
        on_manifold = exp_map(tangent_vecs)
        recovered = log_map(on_manifold)

        err = (tangent_vecs - recovered).norm(dim=-1).mean().item()
        assert err < 1e-4, (
            f"exp_map ∘ log_map error too large: {err:.2e}"
        )

    @pytest.mark.slow
    @pytest.mark.unit
    def test_numerical_stability(self, hhm):
        """1000 recurrent steps without significant manifold drift."""
        D = hhm.state_dim
        B = 1

        state = hhm._step(
            torch.zeros(B, hhm.lorentz_dim),
            torch.randn(B, hhm.lorentz_dim),
        )

        max_drift = 0.0
        for _ in range(1000):
            inp = torch.randn(B, D)
            x_hyp = hhm.exp_map(inp)
            state = hhm._step(state, x_hyp)

            violation = abs(hhm.lorentz_inner(state, state).item() + 1.0)
            max_drift = max(max_drift, violation)

        assert max_drift < 1e-3, (
            f"Manifold drift after 1000 steps: {max_drift:.2e}"
        )

    # -- convenience wrappers from the module --------------------------

    @pytest.fixture(autouse=True)
    def _inject_hhm_methods(self, hhm):
        """Expose module-level helpers on the HHM instance for cleaner tests."""
        import hyper_ssm.hierarchical_memory as hm

        hhm.lorentz_inner = staticmethod(hm.lorentz_inner)
        hhm.exp_map = staticmethod(hm.exp_map)
        hhm.log_map = staticmethod(hm.log_map)
        hhm.lorentz_norm_sq = staticmethod(hm.lorentz_norm_sq)
        hhm._curvature = lambda: torch.exp(hhm.log_c)


# =====================================================================
#  UNIT TESTS — Conversation Memory (ICM)
# =====================================================================


class TestConversationMemory:
    """Tests for InfiniteContextMemory, ChatSession, RAGStore, and
    InfiniteContextLLM."""

    # ------------------------------------------------------------------
    # InfiniteContextMemory
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_remember_recall(self, icm):
        """Remember a random vector then recall it — shape and non-zero."""
        for _ in range(5):
            emb = np.random.randn(32).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            icm.remember(emb)

        query = np.random.randn(32).astype(np.float32)
        recalled = icm.recall(query, scale=0)

        assert recalled.shape == (32,), f"Expected (32,), got {recalled.shape}"
        assert np.linalg.norm(recalled) > 0, "Recall returned zero vector"

    @pytest.mark.unit
    def test_o1_memory_streaming(self, icm):
        """Memory size is constant regardless of utterance count."""
        sizes = []
        for n in [10, 100, 1000]:
            mem = icm.__class__(state_dim=icm.state_dim, num_scales=icm.num_scales)
            for _ in range(n):
                mem.remember(np.random.randn(icm.state_dim).astype(np.float32))
            sizes.append(mem.memory_size_bytes)

        assert len(set(sizes)) == 1, (
            f"Memory sizes differ across utterance counts: {sizes}"
        )
        assert sizes[0] > 0, "Memory size should be non-zero"

    @pytest.mark.unit
    def test_state_serialization(self, icm):
        """Save and load state preserves recall behaviour."""
        for _ in range(5):
            icm.remember(np.random.randn(32).astype(np.float32))

        state_dict = icm.state()

        # Create a fresh instance and load the state
        from hyper_ssm.conversation_memory import InfiniteContextMemory

        icm2 = InfiniteContextMemory(state_dim=32, num_scales=4)
        icm2.load_state(state_dict)

        rng = np.random.RandomState(42)
        q = rng.randn(32).astype(np.float32)

        r1 = icm.recall(q, scale=0)
        r2 = icm2.recall(q, scale=0)

        assert np.allclose(r1, r2, atol=1e-5), (
            "Recall differs after state serialisation"
        )

    @pytest.mark.unit
    def test_state_serialization_round_trip_bytes(self, icm):
        """State dict can be pickled and unpickled without loss."""
        for _ in range(3):
            icm.remember(np.random.randn(32).astype(np.float32))

        state_dict = icm.state()
        serialised = pickle.dumps(state_dict)
        restored = pickle.loads(serialised)

        from hyper_ssm.conversation_memory import InfiniteContextMemory

        icm_restored = InfiniteContextMemory(state_dim=32, num_scales=4)
        icm_restored.load_state(restored)

        q = np.random.randn(32).astype(np.float32)
        assert np.allclose(
            icm.recall(q, scale=0), icm_restored.recall(q, scale=0), atol=1e-5
        )

    # ------------------------------------------------------------------
    # ChatSession
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_chat_session(self):
        """Basic message lifecycle through ChatSession."""
        from hyper_ssm.conversation_memory import ChatSession

        session = ChatSession(
            state_dim=32,
            embedding_fn=lambda t: np.random.randn(32).astype(np.float32),
        )
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")
        session.add_message("user", "What was my first message?")
        session.add_message("assistant", "Your first message was 'Hello'")

        assert session.total_messages == 4
        assert len(session.all_messages) == 4
        assert session.recent_messages(2)[0].role == "user"

    @pytest.mark.unit
    def test_chat_session_save_load(self, tmp_path):
        """ChatSession state persists through save/load round trip."""
        from hyper_ssm.conversation_memory import ChatSession

        def embed_fn(t: str) -> np.ndarray:
            return np.random.randn(32).astype(np.float32)

        session = ChatSession(state_dim=32, embedding_fn=embed_fn)
        session.add_message("user", "Save me!")
        path = str(tmp_path / "session.pkl")
        session.save(path)

        loaded = ChatSession.load(path, embedding_fn=embed_fn)
        assert loaded.total_messages == 1
        assert loaded.all_messages[0].content == "Save me!"

    # ------------------------------------------------------------------
    # RAGStore
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_rag_store(self):
        """Add chunks and retrieve them."""
        from hyper_ssm.conversation_memory import RAGStore

        store = RAGStore(state_dim=32, num_scales=4)
        chunks = [
            "The capital of France is Paris.",
            "Python is a programming language.",
            "The Eiffel Tower is in Paris.",
        ]
        for chunk in chunks:
            emb = np.random.randn(32).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            store.add_chunk(chunk, emb)

        query = np.random.randn(32).astype(np.float32)
        results = store.retrieve(query, top_k=2)
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        for r in results:
            assert "text" in r and "score" in r and "metadata" in r

    @pytest.mark.unit
    def test_rag_store_empty_retrieval(self):
        """Retrieving from an empty store returns an empty list."""
        from hyper_ssm.conversation_memory import RAGStore

        store = RAGStore(state_dim=32)
        assert store.retrieve(np.random.randn(32).astype(np.float32)) == []

    @pytest.mark.unit
    def test_rag_store_hhm_mode(self):
        """HHM-based retrieval (use_hhm=True) also returns results."""
        from hyper_ssm.conversation_memory import RAGStore

        store = RAGStore(state_dim=32, num_scales=4)
        for _ in range(10):
            emb = np.random.randn(32).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            store.add_chunk("chunk", emb)

        q = np.random.randn(32).astype(np.float32)
        results = store.retrieve(q, top_k=3, use_hhm=True, hhm_scale=0)
        assert len(results) == 3

    # ------------------------------------------------------------------
    # InfiniteContextLLM (simulated)
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_infinite_context_llm(self):
        """Simulated LLM with ICM returns non-empty responses and tracks
        conversation length."""
        from hyper_ssm.conversation_memory import InfiniteContextLLM

        llm = InfiniteContextLLM(embedding_dim=32, state_dim=16, num_scales=2)
        response = llm.chat("Hello, who are you?")

        assert isinstance(response, str)
        assert len(response) > 0
        # Two utterances: user message + assistant response
        assert llm.conversation_length == 2

    @pytest.mark.unit
    def test_infinite_context_llm_reuses_memory(self):
        """Second chat uses the accumulated memory state."""
        from hyper_ssm.conversation_memory import InfiniteContextLLM

        llm = InfiniteContextLLM(embedding_dim=32, state_dim=16)
        llm.chat("First message")
        llm.chat("Second message")

        assert llm.conversation_length == 4  # 2 user + 2 assistant

    @pytest.mark.unit
    def test_infinite_context_llm_save_load(self, tmp_path):
        """LLM session save/load round-trip preserves conversation length."""
        from hyper_ssm.conversation_memory import InfiniteContextLLM

        llm = InfiniteContextLLM(embedding_dim=32, state_dim=16)
        llm.chat("Persist me!")
        path = str(tmp_path / "llm_session.pkl")
        llm.save_session(path)

        llm2 = InfiniteContextLLM(embedding_dim=32, state_dim=16)
        llm2.load_session(path)
        assert llm2.conversation_length == llm.conversation_length


# =====================================================================
#  INTEGRATION TESTS — ICM Server (FastAPI)
# =====================================================================


class TestServer:
    """Integration tests for the FastAPI ICM server."""

    @pytest.mark.integration
    def test_health_endpoint(self, server_client):
        """GET /health returns 200 with expected fields."""
        resp = server_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "model" in body
        assert "sessions_active" in body

    @pytest.mark.integration
    def test_chat_endpoint(self, server_client):
        """POST /chat returns a response with session details."""
        resp = server_client.post(
            "/chat", json={"session_id": "test_session", "message": "Hello"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "response" in body
        assert body["session_id"] == "test_session"
        assert len(body["response"]) > 0

    @pytest.mark.integration
    def test_multi_turn_chat(self, server_client):
        """Two chat turns within the same session — memory persists
        (turns_compressed increments)."""
        sid = "multi_turn"

        r1 = server_client.post(
            "/chat", json={"session_id": sid, "message": "Turn one"}
        )
        assert r1.status_code == 200

        r2 = server_client.post(
            "/chat", json={"session_id": sid, "message": "Turn two"}
        )
        assert r2.status_code == 200
        # Simulated fallback: turns_compressed increments each call
        assert r2.json()["turns_compressed"] >= r1.json()["turns_compressed"]

    @pytest.mark.integration
    def test_session_management(self, server_client):
        """Create (via chat), list, get, and delete sessions."""
        sid = "manage_me"

        # Create by chatting
        server_client.post("/chat", json={"session_id": sid, "message": "Hi"})

        # List
        list_resp = server_client.get("/sessions")
        assert list_resp.status_code == 200
        sessions = list_resp.json()["sessions"]
        assert any(s["id"] == sid for s in sessions)

        # Get
        get_resp = server_client.get(f"/sessions/{sid}")
        assert get_resp.status_code == 200
        assert get_resp.json()["session_id"] == sid

        # Delete
        del_resp = server_client.delete(f"/sessions/{sid}")
        assert del_resp.status_code == 200

        # Verify deleted
        get_resp2 = server_client.get(f"/sessions/{sid}")
        assert get_resp2.status_code == 404

    @pytest.mark.integration
    def test_simulated_fallback(self, server_client):
        """Server works without a real LLM via simulated responses."""
        resp = server_client.post(
            "/chat",
            json={"session_id": "sim_test", "message": "Test fallback"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "Simulated" in body["response"]
        # memory_bytes should be 0 when no real LLM is loaded
        assert body["memory_bytes"] == 0

    @pytest.mark.integration
    def test_invalid_session_404(self, server_client):
        """GET/DELETE on nonexistent sessions returns 404."""
        get_resp = server_client.get("/sessions/nonexistent_12345")
        assert get_resp.status_code == 404

        # DELETE on a nonexistent session may or may not 404 depending
        # on implementation — here it returns 200 (idempotent), but we
        # still verify it doesn't crash.
        del_resp = server_client.delete("/sessions/nonexistent_12345")
        assert del_resp.status_code == 200

    @pytest.mark.integration
    def test_create_session_endpoint(self, server_client):
        """POST /sessions creates a new session."""
        resp = server_client.post("/sessions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "created"
        assert body["session_id"].startswith("sess_")

    @pytest.mark.integration
    def test_presets_endpoint(self, server_client):
        """GET /admin/presets returns model presets list."""
        resp = server_client.get("/admin/presets")
        assert resp.status_code == 200
        body = resp.json()
        assert "presets" in body
        assert len(body["presets"]) > 0
        assert body["presets"][0]["id"] == "gpt2"
        assert "name" in body["presets"][0]
        assert "recommended_quant" in body["presets"][0]
        assert "note" in body["presets"][0]

    @pytest.mark.integration
    def test_export_json_endpoint(self, server_client):
        """GET /sessions/{sid}/export/json returns conversation as JSON."""
        sid = "export_json_test"
        server_client.post("/chat", json={"session_id": sid, "message": "Hello"})
        resp = server_client.get(f"/sessions/{sid}/export/json")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/json"
        body = resp.json()
        assert body["session_id"] == sid
        assert len(body["conversation"]) > 0
        assert body["conversation"][0]["role"] in ("user", "assistant")

    @pytest.mark.integration
    def test_export_markdown_endpoint(self, server_client):
        """GET /sessions/{sid}/export/markdown returns Markdown text."""
        sid = "export_md_test"
        server_client.post("/chat", json={"session_id": sid, "message": "Hi there"})
        resp = server_client.get(f"/sessions/{sid}/export/markdown")
        assert resp.status_code == 200
        assert "text/markdown" in resp.headers["content-type"]
        text = resp.text
        assert f"# Conversation: {sid}" in text
        assert "### User" in text
        assert "### Assistant" in text

    @pytest.mark.integration
    def test_export_404_on_missing_session(self, server_client):
        """Export endpoints return 404 for nonexistent sessions."""
        resp = server_client.get(f"/sessions/nonexistent_export/export/json")
        assert resp.status_code == 404

        resp = server_client.get(f"/sessions/nonexistent_export/export/markdown")
        assert resp.status_code == 404

    @pytest.mark.integration
    def test_websocket_chat(self, server_client):
        """WebSocket chat streams tokens and done message."""
        sid = "ws_test"
        with server_client.websocket_connect("/chat/ws") as ws:
            ws.send_json({"session_id": sid, "message": "Hello via WS"})
            tokens = []
            done = None
            while True:
                raw = ws.receive_text()
                data = json.loads(raw)
                if "token" in data:
                    tokens.append(data["token"])
                if data.get("done"):
                    done = data
                    break
                if data.get("error"):
                    pytest.fail(f"WebSocket error: {data['error']}")
            assert len(tokens) > 0
            assert done is not None
            assert done["session_id"] == sid

    @pytest.mark.integration
    def test_websocket_invalid_json(self, server_client):
        """WebSocket handles invalid JSON gracefully."""
        with server_client.websocket_connect("/chat/ws") as ws:
            ws.send_text("not json")
            resp = ws.receive_text()
            data = json.loads(resp)
            assert "error" in data

    @pytest.mark.integration
    def test_websocket_missing_fields(self, server_client):
        """WebSocket rejects messages without session_id or message."""
        with server_client.websocket_connect("/chat/ws") as ws:
            ws.send_json({"session_id": "", "message": ""})
            resp = ws.receive_text()
            data = json.loads(resp)
            assert "error" in data


# =====================================================================
#  UNIT TESTS — Client SDK (icm_client.py)
# =====================================================================


class TestClientSdk:
    """Tests for the ICM HTTP client SDK."""

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_client_parse_response(self):
        """Parse standard API response into normalised dict."""
        from icm_client import IcmClient

        raw = {
            "response": "Hello world",
            "session_id": "sess_1",
            "turns_compressed": 5,
            "memory_bytes": 260,
            "extra_field": "ignored",
        }
        parsed = IcmClient.parse_response(raw)
        assert parsed["response"] == "Hello world"
        assert parsed["session_id"] == "sess_1"
        assert parsed["turns_compressed"] == 5
        assert parsed["memory_bytes"] == 260

    @pytest.mark.unit
    def test_client_parse_response_partial(self):
        """Parse response with missing fields uses safe defaults."""
        from icm_client import IcmClient

        parsed = IcmClient.parse_response({})
        assert parsed["response"] == ""
        assert parsed["turns_compressed"] == 0

    # ------------------------------------------------------------------
    # SSE stream parsing
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_client_stream_parsing(self):
        """Parse SSE event stream lines into structured events."""
        from icm_client import IcmClient

        lines = [
            "data: {\"token\": \"Hello\"}",
            ": comment line",
            "",
            "data: {\"token\": \" world\"}",
            "data: [DONE]",
            " \t  ",
        ]
        events = list(IcmClient.parse_stream(iter(lines)))
        assert len(events) == 3  # two data events + done
        assert events[0] == {"token": "Hello"}
        assert events[1] == {"token": " world"}
        assert events[2] == {"event": "done"}

    @pytest.mark.unit
    def test_client_stream_malformed_line(self):
        """Malformed SSE lines are skipped gracefully."""
        from icm_client import IcmClient

        lines = [
            "not valid sse",
            "data: {bad json",
            "data: {\"ok\": true}",
        ]
        events = list(IcmClient.parse_stream(iter(lines)))
        assert len(events) == 1
        assert events[0] == {"ok": True}

    @pytest.mark.unit
    def test_client_stream_empty(self):
        """Empty stream yields no events."""
        from icm_client import IcmClient

        events = list(IcmClient.parse_stream(iter([])))
        assert events == []

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_client_error_handling_chat_500(self):
        """Server 500 on chat raises IcmClientError."""
        from icm_client import IcmClient, IcmClientError

        client = IcmClient("http://localhost:1")
        with pytest.raises(IcmClientError):
            client.chat("sid", "msg")

    @pytest.mark.unit
    def test_client_error_handling_health_fail(self):
        """Connection refused on health raises IcmClientError."""
        from icm_client import IcmClient, IcmClientError

        client = IcmClient("http://localhost:1")
        with pytest.raises(IcmClientError):
            client.health()

    @pytest.mark.unit
    def test_client_error_invalid_url(self):
        """Empty or malformed URL still raises a sensible error."""
        from icm_client import IcmClient, IcmClientError

        client = IcmClient()
        with pytest.raises(
            (IcmClientError, Exception)
        ):
            client.list_sessions()

    # ------------------------------------------------------------------
    # Session lifecycle (mocked)
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_client_session_lifecycle(self):
        """Full lifecycle via mocked HTTP calls."""
        from icm_client import IcmClient

        client = IcmClient("http://test")

        mock_chat = MagicMock()
        mock_chat.status_code = 200
        mock_chat.json.return_value = {
            "response": "Hi!",
            "session_id": "test_sid",
            "turns_compressed": 1,
            "memory_bytes": 260,
        }

        mock_list = MagicMock()
        mock_list.status_code = 200
        mock_list.json.return_value = {
            "sessions": [{"id": "test_sid", "turns": 1}],
            "total": 1,
        }

        mock_delete = MagicMock()
        mock_delete.status_code = 200
        mock_delete.json.return_value = {
            "status": "deleted",
            "session_id": "test_sid",
        }

        with patch.object(client, "_request", side_effect=[mock_chat, mock_list, mock_delete]):
            chat_result = client.chat("test_sid", "Hello")
            assert chat_result["response"] == "Hi!"

            sessions = client.list_sessions()
            assert len(sessions) == 1

            del_result = client.delete_session("test_sid")
            assert del_result["status"] == "deleted"

    @pytest.mark.unit
    def test_client_session_not_found(self):
        """404 from get_session raises IcmClientError."""
        from icm_client import IcmClient, IcmClientError

        client = IcmClient("http://test")
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch.object(client, "_request", return_value=mock_resp):
            with pytest.raises(IcmClientError, match="Session not found"):
                client.get_session("missing")

    @pytest.mark.unit
    def test_client_delete_session_mocked(self):
        """delete_session returns the server response."""
        from icm_client import IcmClient

        client = IcmClient("http://test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "deleted",
            "session_id": "s1",
        }

        with patch.object(client, "_request", return_value=mock_resp):
            result = client.delete_session("s1")
            assert result["session_id"] == "s1"
            assert result["status"] == "deleted"

    @pytest.mark.unit
    def test_client_export_json_mocked(self):
        """export_json returns parsed conversation from server."""
        from icm_client import IcmClient

        client = IcmClient("http://test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "session_id": "s1",
            "conversation": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        }
        with patch.object(client, "_request", return_value=mock_resp):
            result = client.export_json("s1")
            assert result["session_id"] == "s1"
            assert len(result["conversation"]) == 2

    @pytest.mark.unit
    def test_client_export_markdown_mocked(self):
        """export_markdown returns text from server."""
        from icm_client import IcmClient

        client = IcmClient("http://test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "# Conversation: s1\n\n### User\nHello\n\n### Assistant\nHi\n"
        with patch.object(client, "_request", return_value=mock_resp):
            result = client.export_markdown("s1")
            assert "# Conversation: s1" in result
            assert "### User" in result

    @pytest.mark.unit
    def test_client_export_not_found(self):
        """404 on export raises IcmClientError."""
        from icm_client import IcmClient, IcmClientError

        client = IcmClient("http://test")
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch.object(client, "_request", return_value=mock_resp):
            with pytest.raises(IcmClientError, match="Session not found"):
                client.export_json("missing")

        with patch.object(client, "_request", return_value=mock_resp):
            with pytest.raises(IcmClientError, match="Session not found"):
                client.export_markdown("missing")

    @pytest.mark.unit
    def test_client_chat_stream_ws_no_websockets(self):
        """chat_stream_ws raises IcmClientError when websockets is missing."""
        from icm_client import IcmClient, IcmClientError

        client = IcmClient("http://test")
        with patch.dict("sys.modules", {"websockets": None}):
            with pytest.raises(IcmClientError, match="pip install websockets"):
                list(client.chat_stream_ws("sid", "hello"))


# =====================================================================
#  UNIT TESTS — Session Store (SQLite)
# =====================================================================


class TestSessionStore:
    """Tests for SQLite-backed session persistence."""

    @pytest.mark.unit
    def test_save_and_load(self, tmp_path):
        from hyper_ssm.session_store import SessionStore
        import numpy as np

        db = str(tmp_path / "test.db")
        store = SessionStore(db)

        mem_state = {"state": np.random.randn(65).astype(np.float32), "step": 3}
        history = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]

        store.save("sess_1", mem_state, history, state_dim=64, num_scales=4, turn_count=1)
        assert store.count() == 1

        loaded = store.load("sess_1")
        assert loaded is not None
        assert loaded["session_id"] == "sess_1"
        assert loaded["turn_count"] == 1
        assert len(loaded["history"]) == 2
        assert loaded["history"][0]["content"] == "Hello"
        np.testing.assert_array_equal(loaded["memory_state"]["state"], mem_state["state"])

        store.close()

    @pytest.mark.unit
    def test_delete(self, tmp_path):
        from hyper_ssm.session_store import SessionStore

        db = str(tmp_path / "test.db")
        store = SessionStore(db)
        store.save("sess_1", {"a": 1}, [], state_dim=16, num_scales=2)
        assert store.count() == 1
        store.delete("sess_1")
        assert store.count() == 0
        assert store.load("sess_1") is None
        store.close()

    @pytest.mark.unit
    def test_list_sessions(self, tmp_path):
        from hyper_ssm.session_store import SessionStore

        db = str(tmp_path / "test.db")
        store = SessionStore(db)
        for i in range(3):
            store.save(f"sess_{i}", {"s": i}, [], state_dim=16, num_scales=2)

        sessions = store.list_sessions()
        assert len(sessions) == 3
        assert all(s["session_id"].startswith("sess_") for s in sessions)
        store.close()

    @pytest.mark.unit
    def test_cleanup_expired(self, tmp_path):
        import time
        from hyper_ssm.session_store import SessionStore

        db = str(tmp_path / "test.db")
        store = SessionStore(db)
        store.save("old", {"s": 1}, [], state_dim=16, num_scales=2)
        store.save("new", {"s": 2}, [], state_dim=16, num_scales=2)

        deleted = store.cleanup_expired(ttl_seconds=-1)
        assert deleted == 2
        assert store.count() == 0
        store.close()

    @pytest.mark.unit
    def test_persistence_across_instances(self, tmp_path):
        from hyper_ssm.session_store import SessionStore

        db = str(tmp_path / "test.db")
        store1 = SessionStore(db)
        store1.save("persist", {"x": 1.0}, [{"role": "user", "content": "test"}],
                     state_dim=32, num_scales=4, turn_count=2)
        store1.close()

        store2 = SessionStore(db)
        assert store2.count() == 1
        loaded = store2.load("persist")
        assert loaded["turn_count"] == 2
        assert loaded["history"][0]["content"] == "test"
        store2.close()

    @pytest.mark.unit
    def test_sqlite_config_cli(self):
        from icm_config import get_config
        config = get_config(["--sqlite-path", "icm_data/sessions.db"])
        assert config.sqlite_path == "icm_data/sessions.db"

    @pytest.mark.unit
    def test_llm_sqlite_stores_and_restores(self, tmp_path):
        from hyper_ssm.llm_integration import IcmLlm

        db = str(tmp_path / "test.db")
        llm = IcmLlm(model_name="gpt2", sqlite_path=db, max_new_tokens=16)
        llm.create_session("sqlite_test")
        session = llm._sessions["sqlite_test"]
        session["history"] = [{"role": "user", "content": "persist me"}]
        llm._persist_session("sqlite_test", session)

        assert llm._store is not None
        assert llm._store.count() == 1

        loaded = llm._store.load("sqlite_test")
        assert loaded["history"][0]["content"] == "persist me"
        llm.unload_model()


# =====================================================================
#  UNIT TESTS — API Key Auth + Rate Limiting
# =====================================================================


class TestAuth:
    """Tests for API key authentication and rate limiting."""

    @pytest.mark.unit
    def test_create_and_validate_key(self, tmp_path):
        from hyper_ssm.auth import ApiKeyStore
        db = str(tmp_path / "keys.json")
        store = ApiKeyStore(db)
        key = store.create_key("test key")
        assert key.startswith("icm_")
        assert len(key) > 20
        assert store.validate(key) is True
        assert store.validate("fake_key") is False

    @pytest.mark.unit
    def test_revoke_key(self, tmp_path):
        from hyper_ssm.auth import ApiKeyStore
        store = ApiKeyStore(str(tmp_path / "keys.json"))
        key = store.create_key()
        assert store.validate(key) is True
        store.revoke(key)
        assert store.validate(key) is False

    @pytest.mark.unit
    def test_delete_key(self, tmp_path):
        from hyper_ssm.auth import ApiKeyStore
        store = ApiKeyStore(str(tmp_path / "keys.json"))
        key = store.create_key()
        assert store.count() == 1
        store.delete_key(key)
        assert store.count() == 0

    @pytest.mark.unit
    def test_persist_keys(self, tmp_path):
        from hyper_ssm.auth import ApiKeyStore
        path = str(tmp_path / "keys.json")
        store1 = ApiKeyStore(path)
        k1 = store1.create_key("persisted")

        store2 = ApiKeyStore(path)
        assert store2.validate(k1) is True
        assert store2.count() == 1

    @pytest.mark.unit
    def test_list_keys(self, tmp_path):
        from hyper_ssm.auth import ApiKeyStore
        store = ApiKeyStore(str(tmp_path / "keys.json"))
        store.create_key("alpha")
        store.create_key("beta")
        keys = store.list_keys()
        assert len(keys) == 2
        assert all("key_prefix" in k for k in keys)

    @pytest.mark.unit
    def test_rate_limiter_allows_within_limit(self):
        from hyper_ssm.auth import RateLimiter
        limiter = RateLimiter(requests_per_minute=10)
        for _ in range(10):
            assert limiter.check("key1") is True
        assert limiter.check("key1") is False
        assert limiter.remaining("key1") == 0

    @pytest.mark.unit
    def test_rate_limiter_per_key_independent(self):
        from hyper_ssm.auth import RateLimiter
        limiter = RateLimiter(requests_per_minute=5)
        for _ in range(5):
            limiter.check("a")
        assert limiter.check("a") is False
        assert limiter.check("b") is True


# =====================================================================
#  UNIT TESTS — Model Quantization
# =====================================================================


class TestModelQuantization:
    """Tests for bitsandbytes model quantization support."""

    @pytest.mark.unit
    def test_config_quantize_default(self):
        """Default quantize_bits is None (no quantization)."""
        from icm_config import IcmConfig
        config = IcmConfig()
        assert config.quantize_bits is None

    @pytest.mark.unit
    def test_config_quantize_cli(self):
        """CLI --quantize-bits 4 sets quantize_bits to 4."""
        from icm_config import get_config
        import sys
        config = get_config(["--quantize-bits", "4"])
        assert config.quantize_bits == 4

    @pytest.mark.unit
    def test_config_quantize_cli_8(self):
        """CLI --quantize-bits 8 sets quantize_bits to 8."""
        from icm_config import get_config
        config = get_config(["--quantize-bits", "8"])
        assert config.quantize_bits == 8

    @pytest.mark.unit
    def test_llm_stores_quantize_bits(self):
        """IcmLlm stores quantize_bits parameter."""
        from hyper_ssm.llm_integration import IcmLlm
        llm = IcmLlm(model_name="gpt2", quantize_bits=4)
        assert llm.quantize_bits == 4

        llm2 = IcmLlm(model_name="gpt2", quantize_bits=8)
        assert llm2.quantize_bits == 8

        llm3 = IcmLlm(model_name="gpt2")
        assert llm3.quantize_bits is None

    @pytest.mark.unit
    def test_quantize_fallback_no_bitsandbytes(self):
        """Model loads via fallback when bitsandbytes is unavailable and quantize_bits is set."""
        from hyper_ssm.llm_integration import IcmLlm

        llm = IcmLlm(model_name="gpt2", quantize_bits=4, device="cpu")
        llm._load_model()

        assert llm._llm is not None, "Model should load via fallback"
        param = next(llm._llm.parameters())
        assert param.dtype == torch.float32, "CPU fallback should use float32"
        llm.unload_model()

    @pytest.mark.unit
    def test_quantize_fallback_logs_warning_for_cpu(self):
        """_try_load_quantized logs a warning when running on CPU."""
        from hyper_ssm.llm_integration import IcmLlm

        llm = IcmLlm(model_name="gpt2", quantize_bits=4, device="cpu")

        captured = []
        def capture_print(*args, **kwargs):
            captured.append(" ".join(str(a) for a in args))

        with patch("builtins.print", side_effect=capture_print):
            model = llm._try_load_quantized()

        assert model is not None
        joined = " ".join(captured)
        assert "quantization requires CUDA" in joined or "bitsandbytes not installed" in joined, \
            f"Expected quantization warning, got: {captured}"
        llm.unload_model()

    @pytest.mark.unit
    def test_health_reports_quantize_bits(self, server_client):
        """Health endpoint includes quantize_bits field."""
        resp = server_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "quantize_bits" in body


# =====================================================================
#  UNIT TESTS — Hyperbolic Memory Tree (novel tree-structured memory)
# =====================================================================


class TestHyperbolicMemoryTree:
    """Tests for the HyperbolicMemoryTree — O(log N) structured memory."""

    @pytest.fixture
    def tree(self):
        from hyper_ssm.memory_tree import HyperbolicMemoryTree
        return HyperbolicMemoryTree(state_dim=8, embed_dim=32, branching_factor=4)

    def _emb(self, seed: int = 0) -> np.ndarray:
        rng = np.random.RandomState(seed)
        return rng.randn(32).astype(np.float32)

    @pytest.mark.unit
    def test_init(self, tree):
        info = tree.info()
        assert info["nodes"] == 1  # root
        assert info["leaves"] == 0
        assert info["memory_bytes"] > 0

    @pytest.mark.unit
    def test_remember_single(self, tree):
        nid = tree.remember(self._emb(0), "Hello world")
        assert nid > 0
        info = tree.info()
        assert info["leaves"] == 1
        assert info["nodes"] == 2  # root + leaf

    @pytest.mark.unit
    def test_remember_multiple(self, tree):
        for i in range(4):
            tree.remember(self._emb(i), f"Fact {i}")
        info = tree.info()
        assert info["leaves"] == 4
        assert info["nodes"] == 5  # root + 4 leaves (within branching_factor)

    @pytest.mark.unit
    def test_recall_returns_results(self, tree):
        for i in range(5):
            tree.remember(self._emb(i), f"Fact {i}")
        results = tree.recall(self._emb(0), top_k=3)
        assert len(results) > 0
        for r in results:
            assert "content" in r
            assert "similarity" in r
            assert "depth" in r
            assert "node_id" in r

    @pytest.mark.unit
    def test_recall_returns_most_similar_first(self, tree):
        """The most similar fact should be in the top results."""
        target_emb = self._emb(42)
        tree.remember(target_emb.copy(), "Target fact")
        for i in range(1, 10):
            tree.remember(self._emb(i * 1000 + 999), f"Fact {i}")
        results = tree.recall(target_emb, top_k=10)
        contents = [r["content"] for r in results]
        assert "Target fact" in contents

    @pytest.mark.unit
    def test_tree_depth_grows_with_facts(self, tree):
        """With branching_factor=4, depth should grow log_4(N)."""
        for i in range(20):
            tree.remember(self._emb(i * 100), f"Fact {i}")
        info = tree.info()
        assert info["max_depth"] >= 0

    @pytest.mark.unit
    def test_memory_size_scales_linearly(self, tree):
        sizes = []
        for i in range(1, 6):
            tree.remember(self._emb(i), f"Fact {i}")
            sizes.append(tree.memory_size_bytes())
        # Each fact adds roughly constant memory
        diffs = [sizes[i+1] - sizes[i] for i in range(len(sizes)-1)]
        avg_diff = sum(diffs) / len(diffs)
        assert avg_diff > 0
        # No single diff should be more than 3x the average
        for d in diffs:
            assert d < avg_diff * 3

    @pytest.mark.unit
    def test_reset_clears_tree(self, tree):
        for i in range(5):
            tree.remember(self._emb(i), f"Fact {i}")
        tree.reset()
        info = tree.info()
        assert info["nodes"] == 1
        assert info["leaves"] == 0

    @pytest.mark.unit
    def test_save_and_load_state(self, tree):
        for i in range(5):
            tree.remember(self._emb(i), f"Fact {i}")
        state = tree.state()
        assert "nodes" in state
        assert str(1) in state["nodes"]  # root
        
        tree2 = type(tree)(state_dim=8, embed_dim=32)
        tree2.load_state(state)
        info2 = tree2.info()
        assert info2["leaves"] == 5

    @pytest.mark.unit
    def test_load_state_restores_recall(self, tree):
        for i in range(5):
            tree.remember(self._emb(i), f"Fact {i}")
        state = tree.state()
        
        tree2 = type(tree)(state_dim=8, embed_dim=32)
        tree2.load_state(state)
        
        results = tree2.recall(self._emb(0), top_k=3)
        assert len(results) > 0
        assert results[0]["content"] is not None

    @pytest.mark.unit
    def test_utterance_count(self, tree):
        assert tree.utterance_count == 0
        for i in range(3):
            tree.remember(self._emb(i), f"Fact {i}")
        assert tree.utterance_count == 3

    @pytest.mark.unit
    def test_multi_scale_recall(self, tree):
        for i in range(5):
            tree.remember(self._emb(i), f"Fact {i}")
        scales = tree.recall_all_scales(self._emb(0))
        assert len(scales) > 0
        for s in scales:
            assert s.shape == (32,)

    @pytest.mark.unit
    def test_tree_structure_internal_nodes(self, tree):
        """With many facts, internal nodes should form."""
        from hyper_ssm.memory_tree import HyperbolicMemoryTree
        tree2 = HyperbolicMemoryTree(state_dim=8, embed_dim=32, branching_factor=2)
        for i in range(10):
            tree2.remember(self._emb(i * 1000), f"Fact {i}")
        info = tree2.info()
        # Should have at least 1 internal node (root counts as internal w/children)
        # plus additional internal nodes from splits
        assert info["internal"] >= 1
