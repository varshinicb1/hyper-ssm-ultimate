"""
Infinite Context Memory — FastAPI Server
Serves a single shared IcmLlm instance across all sessions via REST API.
"""

import sys, os, time, pickle, logging, threading, asyncio, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import Response

from icm_config import get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("icm_server")

_cfg = None
_cfg_lock = threading.Lock()

def _get_config():
    global _cfg
    if _cfg is None:
        with _cfg_lock:
            if _cfg is None:
                _cfg = get_config()
                logging.getLogger("icm_server").setLevel(getattr(logging, _cfg.log_level))
    return _cfg

# ---------------------------------------------------------------------------
# Shared LLM backend — loaded in background so server always responds fast
# ---------------------------------------------------------------------------
_llm_instance = None
_llm_instance_lock = threading.Lock()
_llm_loading = False
_llm_ready = threading.Event()

try:
    from hyper_ssm.llm_integration import IcmLlm as LlmBackend
    LLM_BACKEND_AVAILABLE = True
except ImportError:
    LlmBackend = None
    LLM_BACKEND_AVAILABLE = False

# Only load the GPU model when CUDA is actually available
_HAS_CUDA = False
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    pass

def get_llm():
    global _llm_instance, _llm_loading
    if not _HAS_CUDA:
        return None  # CPU: use memory fallback instead
    if _llm_instance is None and not _llm_loading:
        with _llm_instance_lock:
            if _llm_instance is None and not _llm_loading:
                _llm_loading = True
                cfg = _get_config()
                logger.info(f"Loading GPU model in background (model={cfg.model_name})...")
                def _load():
                    global _llm_instance
                    try:
                        instance = LlmBackend(
                            model_name=cfg.model_name,
                            embedder_name=cfg.embedder_name,
                            auto_save_dir=cfg.save_dir,
                            quantize_bits=cfg.quantize_bits,
                            sqlite_path=cfg.sqlite_path,
                        )
                        with _llm_instance_lock:
                            _llm_instance = instance
                        logger.info("IcmLlm ready — GPU model loaded")
                    except Exception as e:
                        logger.warning(f"Failed to load GPU model: {e}")
                    finally:
                        _llm_ready.set()
                thread = threading.Thread(target=_load, daemon=True)
                thread.start()
    if _llm_ready.is_set():
        return _llm_instance
    return None

# ---------------------------------------------------------------------------
# Session tracking
# ---------------------------------------------------------------------------
_session_meta: Dict[str, dict] = {}
_session_lock = threading.Lock()

def _ensure_session(session_id: str):
    with _session_lock:
        max_sessions = _get_config().max_sessions
        if len(_session_meta) >= max_sessions and session_id not in _session_meta:
            oldest = min(_session_meta, key=lambda k: _session_meta[k]["last_active"])
            del _session_meta[oldest]
        if session_id not in _session_meta:
            _session_meta[session_id] = {"last_active": time.time(), "created_at": time.time()}
        _session_meta[session_id]["last_active"] = time.time()

def _get_session_info(session_id: str) -> Optional[dict]:
    with _session_lock:
        meta = _session_meta.get(session_id)
        if meta is None:
            return None
        meta["last_active"] = time.time()
        return dict(meta)

def _delete_session(session_id: str):
    with _session_lock:
        _session_meta.pop(session_id, None)

# ---------------------------------------------------------------------------
# Memory-powered fallback (always works, no GPU needed)
# ---------------------------------------------------------------------------
_mem_sessions = {}
_mem_sessions_lock = threading.Lock()
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _embedder = None
    return _embedder

def _memory_chat(session_id: str, message: str) -> str:
    with _mem_sessions_lock:
        if session_id not in _mem_sessions:
            from hyper_ssm.conversation_memory import InfiniteContextMemory
            _mem_sessions[session_id] = {
                "memory": InfiniteContextMemory(embedding_dim=384, state_dim=64, num_scales=4),
                "history": [],
            }
        entry = _mem_sessions[session_id]

    embedder = _get_embedder()
    if embedder is not None:
        emb = embedder.encode(message, convert_to_numpy=True)
    else:
        emb = np.random.randn(384).astype(np.float32)

    entry["memory"].remember(emb)
    entry["history"].append({"role": "user", "content": message})

    recalled = entry["memory"].recall_all_scales(emb)
    mem_info = entry["memory"].info()

    # Build a response that demonstrates memory recall
    turns = len(entry["history"]) // 2 + 1
    if turns == 1:
        response = f"I hear you. [ICM memory active: {mem_info['memory_bytes']}B, turn 1]"
    else:
        # Show that something was recalled from previous turns
        prev = entry["history"][-3]["content"] if len(entry["history"]) >= 3 else ""
        recall_note = f" (recalled {len(recalled)} scales)" if len(recalled) > 0 else ""
        response = (
            f"I remember our conversation. Earlier you said: \"{prev[:80]}\""
            f"{recall_note}. [ICM: {mem_info['memory_bytes']}B, {turns} turns compressed]"
        )

    entry["history"].append({"role": "assistant", "content": response})
    return response, entry["memory"], entry["history"]

# ---------------------------------------------------------------------------
# Lifespan handler (replaces deprecated on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = _get_config()
    logger.info(f"ICM Server | model={cfg.model_name} embedding={cfg.embedder_name} max_sessions={cfg.max_sessions} quantize={cfg.quantize_bits} sqlite={cfg.sqlite_path}")
    if not _HAS_CUDA:
        logger.info("CPU mode — using memory-powered fallback (instant responses, no GPU needed)")
    elif not LLM_BACKEND_AVAILABLE:
        logger.warning("No LLM backend — all responses simulated")
    cleanup_task = asyncio.create_task(_periodic_cleanup())
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

async def _periodic_cleanup():
    while True:
        await asyncio.sleep(60)
        session_ttl = _get_config().session_ttl
        with _session_lock:
            expired = [sid for sid, m in _session_meta.items()
                      if time.time() - m["last_active"] > session_ttl]
            for sid in expired:
                del _session_meta[sid]
        if expired:
            logger.info(f"Cleaned {len(expired)} expired sessions")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Infinite Context Memory API", version="1.0.0",
              description="O(1) hyperbolic memory for LLMs served via REST.",
              lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Static files mounted from {static_dir}")

# ---------------------------------------------------------------------------
# API Key Auth + Rate Limiting middleware
# ---------------------------------------------------------------------------
_auth_store = None
_auth_lock = threading.Lock()
_rate_limiter = None

def _get_auth():
    global _auth_store, _rate_limiter
    cfg = _get_config()
    if not cfg.auth_enabled:
        return None, None
    if _auth_store is None:
        with _auth_lock:
            if _auth_store is None:
                from hyper_ssm.auth import ApiKeyStore, RateLimiter
                keys_path = cfg.auth_keys_path or os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "icm_keys.json"
                )
                _auth_store = ApiKeyStore(keys_path)
                _rate_limiter = RateLimiter(requests_per_minute=cfg.rate_limit_rpm)
                if _auth_store.count() == 0:
                    default_key = _auth_store.create_key("default admin key")
                    logger.warning(f"No API keys found — created default key: {default_key}")
                    logger.warning(f"  Use header: Authorization: Bearer {default_key}")
    return _auth_store, _rate_limiter

PUBLIC_PATHS = {"/", "/health", "/admin", "/admin.html", "/docs", "/openapi.json",
                "/redoc", "/static", "/favicon.ico"}

@app.middleware("http")
async def auth_middleware(request, call_next):
    cfg = _get_config()
    path = request.url.path

    if not cfg.auth_enabled or any(path.startswith(p) for p in PUBLIC_PATHS):
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "")
    api_key = request.headers.get("X-API-Key", "")

    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]
    elif auth_header.startswith("ApiKey "):
        api_key = auth_header[7:]

    if not api_key:
        return HTMLResponse(
            json.dumps({"error": "Missing API key. Use Authorization: Bearer <key>"}),
            status_code=401,
            headers={"Content-Type": "application/json",
                     "WWW-Authenticate": "Bearer"},
        )

    store, limiter = _get_auth()
    if store is None:
        return await call_next(request)

    if not store.validate(api_key):
        return HTMLResponse(
            json.dumps({"error": "Invalid API key"}),
            status_code=401,
            headers={"Content-Type": "application/json"},
        )

    if limiter and not limiter.check(api_key):
        remaining = limiter.remaining(api_key)
        return HTMLResponse(
            json.dumps({"error": "Rate limit exceeded", "retry_after_seconds": 60}),
            status_code=429,
            headers={"Content-Type": "application/json",
                     "X-RateLimit-Remaining": str(remaining)},
        )

    return await call_next(request)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ModelSwitchRequest(BaseModel):
    model_name: str
    quantize_bits: Optional[int] = None

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/health")
async def get_health():
    cfg = _get_config()
    with _session_lock:
        active = len(_session_meta)
    mode = "gpu" if _HAS_CUDA else "cpu-memory-fallback"
    return {"status": "ok", "model": cfg.model_name, "embedding": cfg.embedder_name,
            "llm_available": LLM_BACKEND_AVAILABLE, "mode": mode,
            "sessions_active": active, "max_sessions": cfg.max_sessions,
            "session_ttl": cfg.session_ttl, "quantize_bits": cfg.quantize_bits,
            "sqlite_path": cfg.sqlite_path, "auth_enabled": cfg.auth_enabled}

# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

MODEL_PRESETS = [
    {"id": "gpt2", "name": "GPT-2 (124M)", "recommended_quant": None, "note": "Fastest, good for testing"},
    {"id": "distilgpt2", "name": "DistilGPT-2 (82M)", "recommended_quant": None, "note": "Smallest, fastest"},
    {"id": "Qwen/Qwen2.5-0.5B", "name": "Qwen 2.5 (0.5B)", "recommended_quant": 4, "note": "Good balance, 4-bit recommended"},
    {"id": "Qwen/Qwen2.5-1.5B", "name": "Qwen 2.5 (1.5B)", "recommended_quant": 4, "note": "Smarter, needs 4-bit for 8GB VRAM"},
    {"id": "microsoft/Phi-3-mini-4k-instruct", "name": "Phi-3 Mini (3.8B)", "recommended_quant": 4, "note": "Strong instruct model, 4-bit"},
    {"id": "microsoft/Phi-3.5-mini-instruct", "name": "Phi-3.5 Mini (3.8B)", "recommended_quant": 4, "note": "Updated Phi-3, 4-bit"},
    {"id": "google/gemma-2-2b", "name": "Gemma 2 (2B)", "recommended_quant": 4, "note": "Google's efficient model, 4-bit"},
    {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "name": "TinyLlama (1.1B)", "recommended_quant": None, "note": "Good CPU model, no quant needed"},
    {"id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "name": "SmolLM2 (1.7B)", "recommended_quant": None, "note": "Latest small instruct model"},
]

@app.post("/sessions")
async def create_session():
    import uuid
    sid = f"sess_{uuid.uuid4().hex[:12]}"
    _ensure_session(sid)
    llm = get_llm()
    if llm is not None:
        try:
            llm.create_session(sid)
        except Exception:
            pass
    return {"session_id": sid, "status": "created"}

@app.get("/sessions")
async def list_sessions():
    cfg = _get_config()
    with _session_lock:
        sessions_data = dict(_session_meta)
    llm_ref = get_llm()
    llm_sessions = set(llm_ref._sessions.keys()) if llm_ref is not None else set()
    mem_sessions = set(_mem_sessions.keys()) if _mem_sessions else set()
    infos = []
    for sid, m in sessions_data.items():
        turns = 0
        if sid in llm_sessions:
            history = llm_ref._sessions[sid].get("history", [])
            turns = len(history) // 2
        elif sid in mem_sessions:
            history = _mem_sessions[sid]["history"]
            turns = len(history) // 2
        infos.append({"id": sid, "turns": turns,
                      "last_active": m["last_active"], "created_at": m["created_at"]})
    total = len(infos)
    if llm_ref is not None and llm_ref._store:
        total = llm_ref._store.count()
    return {"sessions": infos, "total": total, "max": cfg.max_sessions}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    _delete_session(session_id)
    with _mem_sessions_lock:
        _mem_sessions.pop(session_id, None)
    try:
        llm = get_llm()
        if llm is not None and session_id in llm._sessions:
            llm.delete_session(session_id)
    except Exception:
        pass
    return {"status": "deleted", "session_id": session_id}

@app.post("/chat")
async def chat(req: ChatRequest):
    _ensure_session(req.session_id)
    llm = get_llm()
    if llm is not None:
        async def _do_chat():
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, llm.chat, req.session_id, req.message)
        try:
            response = await _do_chat()
        except Exception as exc:
            logger.error(f"Chat error [{req.session_id}]: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))
        session = llm._sessions.get(req.session_id, {})
        memory = session.get("memory")
        mem_bytes = memory.memory_size_bytes if memory else 0
        turns = llm.conversation_length // 2 if hasattr(llm, "conversation_length") else 0
        return {"response": response, "session_id": req.session_id,
                "turns_compressed": turns, "memory_bytes": mem_bytes}
    # Fallback: memory-powered response (always instant)
    response, memory, history = _memory_chat(req.session_id, req.message)
    mem_bytes = memory.memory_size_bytes if memory else 0
    turns = len(history) // 2
    return {"response": response, "session_id": req.session_id,
            "turns_compressed": turns, "memory_bytes": mem_bytes}

@app.get("/chat/stream")
async def chat_stream(session_id: str, message: str):
    _ensure_session(session_id)
    async def event_generator():
        llm = get_llm()
        if llm is not None:
            loop = asyncio.get_running_loop()
            q = asyncio.Queue()
            def _run_stream():
                try:
                    for token in llm.chat_stream(session_id, message):
                        loop.call_soon_threadsafe(q.put_nowait, ("token", token))
                except Exception as e:
                    loop.call_soon_threadsafe(q.put_nowait, ("error", str(e)))
                loop.call_soon_threadsafe(q.put_nowait, ("done", None))
            thread = threading.Thread(target=_run_stream, daemon=True)
            thread.start()
            while True:
                event_type, data = await q.get()
                if event_type == "error":
                    yield f"event: error\ndata: {json.dumps({'error': data})}\n\n"
                    return
                if event_type == "done":
                    break
                yield f"event: token\ndata: {json.dumps({'token': data, 'session_id': session_id})}\n\n"
            session = llm._sessions.get(session_id, {})
            memory = session.get("memory")
            mem_bytes = memory.memory_size_bytes if memory else 0
            turns = memory._utterance_count // 2 if memory else 0
        else:
            response, memory, history = _memory_chat(session_id, message)
            for word in response.split(" "):
                yield f"event: token\ndata: {json.dumps({'token': word + ' ', 'session_id': session_id})}\n\n"
            mem_bytes = memory.memory_size_bytes if memory else 0
            turns = len(history) // 2
        yield f"event: done\ndata: {json.dumps({'session_id': session_id, 'turns_compressed': turns, 'memory_bytes': mem_bytes})}\n\n"
        yield f"event: done\ndata: {json.dumps({'session_id': session_id, 'turns_compressed': turns, 'memory_bytes': mem_bytes})}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ---------------------------------------------------------------------------
# WebSocket chat endpoint
# ---------------------------------------------------------------------------

@app.websocket("/chat/ws")
async def chat_websocket(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"error": "Invalid JSON"})
                continue

            session_id = data.get("session_id", "")
            message = data.get("message", "")
            if not session_id or not message:
                await ws.send_json({"error": "session_id and message required"})
                continue

            _ensure_session(session_id)
            llm = get_llm()

            if llm is None:
                response, memory, history = _memory_chat(session_id, message)
                for word in response.split(" "):
                    await ws.send_json({"token": word + " ", "session_id": session_id})
                    await asyncio.sleep(0.02)
                mem_bytes = memory.memory_size_bytes if memory else 0
                turns = len(history) // 2
                await ws.send_json({
                    "done": True, "session_id": session_id,
                    "turns_compressed": turns, "memory_bytes": mem_bytes,
                })
                continue
            q = asyncio.Queue()

            def _run():
                try:
                    for token in llm.chat_stream(session_id, message):
                        loop.call_soon_threadsafe(q.put_nowait, ("token", token))
                except Exception as e:
                    loop.call_soon_threadsafe(q.put_nowait, ("error", str(e)))
                loop.call_soon_threadsafe(q.put_nowait, ("done", None))

            loop = asyncio.get_running_loop()
            thread = threading.Thread(target=_run, daemon=True)
            thread.start()

            while True:
                event_type, payload = await q.get()
                if event_type == "error":
                    await ws.send_json({"error": payload, "session_id": session_id})
                    break
                if event_type == "done":
                    break
                await ws.send_json({"token": payload, "session_id": session_id})

            session = llm._sessions.get(session_id, {})
            memory = session.get("memory")
            mem_bytes = memory.memory_size_bytes if memory else 0
            turns = memory._utterance_count // 2 if memory else 0
            await ws.send_json({
                "done": True, "session_id": session_id,
                "turns_compressed": turns, "memory_bytes": mem_bytes,
            })
    except WebSocketDisconnect:
        pass
    except Exception:
        pass

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    meta = _get_session_info(session_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Session not found")
    result = {"session_id": session_id, **meta}
    if LLM_BACKEND_AVAILABLE:
        llm = get_llm()
        if session_id in llm._sessions:
            memory = llm._sessions[session_id]["memory"]
            result["memory_bytes"] = memory.memory_size_bytes
            result["turns"] = llm._sessions[session_id]["memory"]._utterance_count
            try:
                result["manifold_violation"] = float(memory.info()["state_on_manifold"])
            except Exception:
                pass
    return result

# ---------------------------------------------------------------------------
# Conversation export
# ---------------------------------------------------------------------------

def _get_session_history(session_id: str):
    llm = get_llm()
    if llm is not None:
        session = llm._sessions.get(session_id)
        if session is not None:
            return session["history"]
    with _mem_sessions_lock:
        mem_entry = _mem_sessions.get(session_id)
        if mem_entry is not None:
            return mem_entry["history"]
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/sessions/{session_id}/export/json")
async def export_session_json(session_id: str):
    history = _get_session_history(session_id)
    payload = {"session_id": session_id, "conversation": history}
    return Response(
        content=json.dumps(payload, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.json"'},
    )

@app.get("/sessions/{session_id}/export/markdown")
async def export_session_markdown(session_id: str):
    history = _get_session_history(session_id)
    lines = [
        f"# Conversation: {session_id}",
        "",
        f"*Exported at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(time.time()))}*",
        "",
        "---",
        "",
    ]
    for turn in history:
        role = turn["role"].capitalize()
        content = turn["content"]
        lines.append(f"### {role}")
        lines.append("")
        lines.append(content)
        lines.append("")
        lines.append("---")
        lines.append("")
    md = "\n".join(lines)
    return Response(
        content=md,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.md"'},
    )

# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------

@app.get("/admin")
async def get_admin():
    html_path = os.path.join(static_dir, "admin.html")
    if os.path.isfile(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Admin Dashboard</h1><p>admin.html not found</p>")

@app.get("/admin/stats")
async def admin_stats():
    cfg = _get_config()
    stats = {
        "model_name": cfg.model_name,
        "embedder_name": cfg.embedder_name,
        "quantize_bits": cfg.quantize_bits,
        "sqlite_path": cfg.sqlite_path,
        "llm_available": LLM_BACKEND_AVAILABLE,
        "max_sessions": cfg.max_sessions,
        "session_ttl": cfg.session_ttl,
        "state_dim": cfg.state_dim,
        "num_scales": cfg.num_scales,
        "max_new_tokens": cfg.max_new_tokens,
    }
    llm = get_llm()
    if llm is not None:
        stats["total_conversation_utterances"] = llm.conversation_length
        if llm._store:
            stats["sqlite_session_count"] = llm._store.count()
        stats["loaded_model"] = llm.model_name
        stats["device"] = str(llm.device)
    else:
        stats["loaded_model"] = cfg.model_name
        stats["device"] = "CPU (memory-fallback mode — real LLM requires CUDA)"
    with _session_lock:
        stats["in_memory_sessions"] = len(_session_meta)
    try:
        import torch
        if torch.cuda.is_available():
            stats["cuda"] = True
            stats["cuda_device"] = torch.cuda.get_device_name(0)
            stats["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
            stats["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024**2, 1)
        else:
            stats["cuda"] = False
        stats["cuda_device"] = None
    except Exception:
        stats["cuda"] = False
    store, limiter = _get_auth()
    stats["auth_enabled"] = cfg.auth_enabled
    stats["auth_key_count"] = store.count() if store else 0
    stats["rate_limit_rpm"] = cfg.rate_limit_rpm
    return stats

@app.get("/admin/sessions/{session_id}/history")
async def admin_session_history(session_id: str):
    llm = get_llm()
    session = None
    if llm is not None:
        session = llm._sessions.get(session_id)
    if session is None:
        with _mem_sessions_lock:
            mem_entry = _mem_sessions.get(session_id)
        if mem_entry is not None:
            memory = mem_entry["memory"]
            return {
                "session_id": session_id,
                "history": mem_entry["history"],
                "utterance_count": memory._utterance_count,
                "memory_bytes": memory.memory_size_bytes,
            }
        raise HTTPException(status_code=404, detail="Session not found")
    memory = session["memory"]
    return {
        "session_id": session_id,
        "history": session["history"],
        "utterance_count": memory._utterance_count,
        "memory_bytes": memory.memory_size_bytes,
    }

class ModelSwitchRequest(BaseModel):
    model_name: str
    quantize_bits: Optional[int] = None

@app.post("/admin/model")
async def admin_switch_model(req: ModelSwitchRequest):
    if not LLM_BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLM backend not available")
    llm = get_llm()
    try:
        msg = llm.switch_model(req.model_name, quantize_bits=req.quantize_bits)
        return {"status": "ok", "message": msg}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/presets")
async def admin_presets():
    return {"presets": MODEL_PRESETS}

# ---------------------------------------------------------------------------
# Auth management endpoints (always accessible without auth key)
# ---------------------------------------------------------------------------

@app.get("/admin/keys")
async def admin_list_keys():
    store, _ = _get_auth()
    if store is None:
        return {"keys": [], "enabled": False}
    return {"keys": store.list_keys(), "enabled": True, "total": store.count()}

class CreateKeyRequest(BaseModel):
    description: str = ""

@app.post("/admin/keys")
async def admin_create_key(req: CreateKeyRequest):
    store, _ = _get_auth()
    if store is None:
        raise HTTPException(status_code=503, detail="Auth not enabled")
    key = store.create_key(description=req.description)
    return {"key": key, "description": req.description, "message": "Store this key — it won't be shown again"}

@app.delete("/admin/keys/{key_id}")
async def admin_delete_key(key_id: str):
    store, _ = _get_auth()
    if store is None:
        raise HTTPException(status_code=503, detail="Auth not enabled")
    if store.delete_key(key_id):
        return {"status": "deleted", "key_id": key_id}
    raise HTTPException(status_code=404, detail="Key not found")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import uvicorn
    cfg = get_config()
    uvicorn.run(app, host=cfg.host, port=cfg.port)

if __name__ == "__main__":
    main()
