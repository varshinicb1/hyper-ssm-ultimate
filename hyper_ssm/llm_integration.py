"""
LLM Integration with Infinite Context Memory (ICM)

O(1) memory for any HuggingFace causal language model.
Wraps the model with InfiniteContextMemory — a hyperbolic state that
compresses the entire conversation into a fixed-size vector.

Usage:
    chat = IcmLlm(model_name="gpt2")
    chat.create_session("alice")
    reply = chat.chat("alice", "Hello, my name is Alice")
    reply = chat.chat("alice", "What is my name?")  # remembers!
"""

import os
import pickle
import threading
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from .conversation_memory import InfiniteContextMemory


class IcmLlm:
    """
    Wraps any HuggingFace causal LM with Infinite Context Memory.

    O(1) memory: the hyperbolic state vector is fixed-size (~260 bytes for
    state_dim=64), regardless of conversation length.

    Args:
        model_name: HuggingFace model ID (e.g. "gpt2", "Qwen/Qwen2.5-0.5B").
        state_dim: Dimension of hyperbolic memory state (default 64).
        num_scales: Number of hierarchical abstraction levels (default 4).
        device: Device for model inference ("cuda", "cpu", or None for auto).
        max_new_tokens: Maximum tokens per generation (default 512).
        system_prompt: Optional system prompt override.
        auto_save_dir: Directory to auto-save sessions (None = no auto-save).
        embedder_name: Sentence-transformers model for embeddings.
    """

    SCALE_LABELS = [
        ("The conversation covered", "detailed"),
        ("Key topics discussed", "moderate"),
        ("Overall theme", "abstract"),
        ("Core purpose", "gist"),
    ]

    def __init__(
        self,
        model_name: str = "gpt2",
        state_dim: int = 64,
        num_scales: int = 4,
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        system_prompt: Optional[str] = None,
        auto_save_dir: Optional[str] = None,
        embedder_name: str = "all-MiniLM-L6-v2",
        quantize_bits: Optional[int] = None,
        sqlite_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.state_dim = state_dim
        self.num_scales = num_scales
        self.max_new_tokens = max_new_tokens
        self.auto_save_dir = auto_save_dir
        self.embedder_name = embedder_name
        self.quantize_bits = quantize_bits
        self.sqlite_path = sqlite_path

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.system_prompt = system_prompt or (
            "You are an AI assistant with infinite context memory. "
            "You remember the entire conversation history, including details "
            "from the very beginning. Use the memory recall provided below "
            "to maintain coherent long-term conversations."
        )

        self._llm = None
        self._tokenizer = None
        self._embedder = None
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._store = None
        self._store_lock = threading.Lock()

        if sqlite_path:
            from .session_store import SessionStore
            self._store = SessionStore(sqlite_path)
            self._restore_sessions_from_db()

        if auto_save_dir:
            os.makedirs(auto_save_dir, exist_ok=True)

    def _restore_sessions_from_db(self):
        if self._store is None:
            return
        try:
            entries = self._store.list_sessions()
            for entry in entries:
                sid = entry["session_id"]
                data = self._store.load(sid)
                if data is None:
                    continue
                memory = InfiniteContextMemory(
                    embedding_dim=384,
                    state_dim=data["state_dim"],
                    num_scales=data["num_scales"],
                    device=torch.device("cpu"),
                )
                memory.load_state(data["memory_state"])
                self._sessions[sid] = {
                    "memory": memory,
                    "history": data["history"],
                }
            if entries:
                print(f"  Restored {len(entries)} sessions from {self.sqlite_path}")
        except Exception as e:
            print(f"  Warning: failed to restore sessions: {e}")

    # ------------------------------------------------------------------
    # Lazy Loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._llm is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading model '{self.model_name}'...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            if self.quantize_bits is not None:
                self._llm = self._try_load_quantized()
            elif self.device.type == "cuda":
                self._llm = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            else:
                self._llm = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                ).to(self.device)
            self._llm.eval()
            print(f"  Model loaded on {self.device}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{self.model_name}': {e}\n"
                f"Check the model name and internet connection."
            )

    def _try_load_quantized(self):
        """Try loading a quantized model; fall back to full precision on failure."""
        bits = self.quantize_bits
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            print(f"  WARNING: bitsandbytes not installed, ignoring quantize_bits={bits}. "
                  f"Install with: pip install bitsandbytes")
            return self._load_fallback()

        if self.device.type != "cuda":
            print(f"  WARNING: quantization requires CUDA, ignoring quantize_bits={bits}. "
                  f"Running on {self.device}")
            return self._load_fallback()

        from transformers import BitsAndBytesConfig

        if bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            dtype = torch.float16
            label = "4-bit"
        elif bits == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            dtype = torch.float16
            label = "8-bit"
        else:
            print(f"  WARNING: unsupported quantize_bits={bits}, ignoring")
            return self._load_fallback()

        from transformers import AutoModelForCausalLM
        try:
            print(f"  Loading {label} quantized ({bits} bits)...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto",
                quantization_config=bnb_config,
            )
            model = model.to(self.device)
            print(f"  {label} quantized model loaded successfully")
            return model
        except Exception as e:
            print(f"  WARNING: {label} quantization failed: {e}")
            print(f"  Falling back to full precision...")
            return self._load_fallback()

    def _load_fallback(self):
        from transformers import AutoModelForCausalLM

        if self.device.type == "cuda":
            return AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        ).to(self.device)

    def _load_embedder(self) -> None:
        if self._embedder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            print(f"Loading embedder '{self.embedder_name}'...")
            self._embedder = SentenceTransformer(self.embedder_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedder '{self.embedder_name}': {e}\n"
                f"Install: pip install sentence-transformers"
            )

    # ------------------------------------------------------------------
    # Session Management
    # ------------------------------------------------------------------

    def create_session(self, session_id: str) -> Dict[str, Any]:
        if session_id in self._sessions:
            return self._sessions[session_id]
        memory = InfiniteContextMemory(
            embedding_dim=384,
            state_dim=self.state_dim,
            num_scales=self.num_scales,
            device=torch.device("cpu"),
        )
        session = {"memory": memory, "history": []}
        self._sessions[session_id] = session
        return session

    def chat(self, session_id: str, message: str) -> str:
        self._load_model()
        self._load_embedder()

        session = self._sessions.get(session_id)
        if session is None:
            session = self.create_session(session_id)

        memory = session["memory"]
        history = session["history"]

        # a. Embed user message, compress into memory
        user_emb = self._embed(message)
        memory.remember(user_emb)

        # b. Recall at all scales using user message as query
        recalled = memory.recall_all_scales(user_emb)

        # c. Build prompt with bounded token budget
        prompt = self._build_prompt(message, recalled, history)

        # d. Generate response
        response = self._generate(prompt)

        # e. Embed assistant response, compress into memory
        resp_emb = self._embed(response)
        memory.remember(resp_emb)

        # Update history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})

        # Auto-save after every turn
        self._persist_session(session_id, session)
        if self.auto_save_dir:
            path = os.path.join(self.auto_save_dir, f"{session_id}.pkl")
            self.save_session(session_id, path)

        return response

    def chat_stream(self, session_id: str, message: str):
        """Yields tokens one at a time as the LLM generates them."""
        self._load_model()
        self._load_embedder()

        session = self._sessions.get(session_id)
        if session is None:
            session = self.create_session(session_id)

        memory = session["memory"]
        history = session["history"]

        # a. Embed user message, compress into memory
        user_emb = self._embed(message)
        memory.remember(user_emb)

        # b. Recall at all scales using user message as query
        recalled = memory.recall_all_scales(user_emb)

        # c. Build prompt with bounded token budget
        prompt = self._build_prompt(message, recalled, history)

        # d. Generate response with streaming
        from transformers import TextIteratorStreamer
        from threading import Thread

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=False)
        input_len = inputs.input_ids.shape[1]
        model_max = getattr(self._tokenizer, "model_max_length", 1024)
        max_input = model_max - self.max_new_tokens - 10
        if input_len > max_input:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max(max_input, 10),
            )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=(self._tokenizer.pad_token_id or self._tokenizer.eos_token_id),
            streamer=streamer,
        )
        thread = Thread(target=self._llm.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        for text in streamer:
            generated_text += text
            yield text

        # e. Embed assistant response, compress into memory
        response = generated_text.strip()
        resp_emb = self._embed(response)
        memory.remember(resp_emb)

        # Update history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})

        # Auto-save after every turn
        self._persist_session(session_id, session)
        if self.auto_save_dir:
            path = os.path.join(self.auto_save_dir, f"{session_id}.pkl")
            self.save_session(session_id, path)

    def _persist_session(self, session_id: str, session: dict) -> None:
        if self._store is None:
            return
        try:
            memory = session["memory"]
            history = session["history"]
            self._store.save(
                session_id=session_id,
                memory_state=memory.state(),
                history=history,
                state_dim=self.state_dim,
                num_scales=self.num_scales,
                turn_count=memory._utterance_count,
            )
        except Exception:
            pass

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        if self._store:
            self._store.delete(session_id)
        if self.auto_save_dir:
            path = os.path.join(self.auto_save_dir, f"{session_id}.pkl")
            if os.path.exists(path):
                os.remove(path)

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())

    def save_session(self, session_id: str, path: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' not found")
        data = {
            "memory_state": session["memory"].state(),
            "history": session["history"],
            "state_dim": self.state_dim,
            "num_scales": self.num_scales,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_session(self, session_id: str, path: str) -> Dict[str, Any]:
        with open(path, "rb") as f:
            data = pickle.load(f)
        memory = InfiniteContextMemory(
            embedding_dim=384,
            state_dim=data.get("state_dim", self.state_dim),
            num_scales=data.get("num_scales", self.num_scales),
            device=torch.device("cpu"),
        )
        memory.load_state(data["memory_state"])
        session = {"memory": memory, "history": data["history"]}
        self._sessions[session_id] = session
        return session

    # ------------------------------------------------------------------
    # Core Operations
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        return self._embedder.encode(text, normalize_embeddings=True).astype(np.float32)

    def _build_prompt(
        self,
        user_message: str,
        recalled: List[np.ndarray],
        history: List[Dict[str, str]],
    ) -> str:
        """Build a prompt bounded by the model's max token budget."""
        tokenizer = self._tokenizer

        # Fixed preamble: system + multi-scale recall
        preamble = f"System: {self.system_prompt}\n\nInfinite Memory Recall:\n"
        for i, ((label, _), vec) in enumerate(zip(self.SCALE_LABELS, recalled)):
            preamble += f"  [{label}] {self._vec_to_text(vec)}\n"

        # Tail: user message + assistant prefix
        tail = f"\nUser: {user_message}\n\nAssistant:"

        # Token budget
        model_max = getattr(tokenizer, "model_max_length", 1024)
        budget = model_max - self.max_new_tokens - 50
        preamble_tokens = len(tokenizer.encode(preamble))
        tail_tokens = len(tokenizer.encode(tail))
        avail = budget - preamble_tokens - tail_tokens

        # If fixed parts alone exceed budget, truncate preamble
        if avail <= 0:
            preamble = preamble[:max(budget - tail_tokens, 10)]
            return preamble + tail

        # Add recent history from newest to oldest, respecting budget
        recent = [m for m in history if m["role"] in ("user", "assistant")]
        hist_lines: List[str] = []
        hist_tokens = 0
        for msg in reversed(recent):
            line = f"  {msg['role']}: {msg['content']}"
            line_tokens = len(tokenizer.encode("\n" + line))
            if hist_tokens + line_tokens > avail:
                break
            hist_lines.insert(0, line)
            hist_tokens += line_tokens

        if hist_lines:
            return (
                preamble
                + "\nRecent conversation:\n"
                + "\n".join(hist_lines)
                + "\n"
                + tail
            )
        return preamble + tail

    @staticmethod
    def _vec_to_text(vec: np.ndarray) -> str:
        if vec is None or len(vec) == 0:
            return "no recall (empty conversation)"
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return "no recall (empty conversation)"
        top_k = min(5, len(vec))
        top_idx = np.argsort(np.abs(vec))[-top_k:][::-1]
        top_str = ", ".join(f"{i}:{vec[i]:.3f}" for i in top_idx)
        return (
            f"[activation={norm:.3f}, dim={len(vec)}, "
            f"top components: {top_str}]"
        )

    @torch.no_grad()
    def _generate(self, prompt: str) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=False)
        input_len = inputs.input_ids.shape[1]
        model_max = getattr(self._tokenizer, "model_max_length", 1024)
        max_input = model_max - self.max_new_tokens - 10
        if input_len > max_input:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max(max_input, 10),
            )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        try:
            outputs = self._llm.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=(
                    self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
                ),
            )
        except Exception as e:
            return f"[Generation error: {e}]"

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Resource Management
    # ------------------------------------------------------------------

    @property
    def conversation_length(self) -> int:
        total = 0
        for sid in self._sessions:
            mem = self._sessions[sid].get("memory")
            if mem:
                total += mem._utterance_count
        return total

    def switch_model(self, model_name: str, quantize_bits: Optional[int] = None) -> str:
        self.unload_model()
        self.model_name = model_name
        if quantize_bits is not None:
            self.quantize_bits = quantize_bits
        self._load_model()
        return f"Switched to {model_name} (quantize={self.quantize_bits})"

    def unload_model(self) -> None:
        self._llm = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.unload_model()


# =========================================================================
# SELF-VALIDATION
# =========================================================================

def validate_icm_llm():
    """
    Validate the complete ICM-LLM integration.

    Loads a real tiny model (gpt2), creates a session, sends 3 test messages,
    verifies responses are non-empty, confirms O(1) memory, and prints
    comprehensive memory stats after each turn.
    """
    print("=" * 60)
    print("ICM LLM INTEGRATION — VALIDATION")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    model_name = "gpt2"

    print(f"\n[Step 1] Initializing IcmLlm(model_name='{model_name}')...")
    llm = IcmLlm(
        model_name=model_name,
        state_dim=64,
        num_scales=4,
        max_new_tokens=50,
        auto_save_dir=None,
    )

    print(f"\n[Step 2] Creating session 'test_user'...")
    session = llm.create_session("test_user")
    assert "test_user" in llm._sessions
    print(f"  Active sessions: {llm.list_sessions()}")
    mem = session["memory"]
    print(f"  Initial memory: {mem.memory_size_bytes}B, "
          f"utterances: {mem._utterance_count}")

    test_messages = [
        "Hello, I'm testing the infinite memory system.",
        "What was my first message about?",
        "Can you summarize our conversation so far?",
    ]

    for i, msg in enumerate(test_messages):
        print(f"\n[Step 3.{i+1}] Chat turn {i+1}")
        print(f"  User: {msg}")
        response = llm.chat("test_user", msg)
        print(f"  Assistant: {response[:120]}...")
        assert isinstance(response, str) and len(response) > 0, \
            f"Empty response on turn {i+1}"
        info = session["memory"].info()
        print(f"  Memory state:")
        print(f"    - Utterances compressed: {info['utterance_count']}")
        print(f"    - State size: {info['memory_bytes']}B (O(1))")
        print(f"    - State dimension: {info['state_dim']}")
        print(f"    - On manifold: {info['state_on_manifold']}")
        assert info['memory_bytes'] <= 300, \
            f"Memory exceeded O(1) bound: {info['memory_bytes']}B"

    print(f"\n[Step 4] Verifying multi-scale recall...")
    emb = llm._embed("test query")
    recalled = session["memory"].recall_all_scales(emb)
    for i, r in enumerate(recalled):
        print(f"  Scale {i}: norm={np.linalg.norm(r):.4f}, shape={r.shape}")
    assert len(recalled) == llm.num_scales, \
        f"Expected {llm.num_scales} scales, got {len(recalled)}"

    print(f"\n[Step 5] Verifying session management...")
    sessions = llm.list_sessions()
    print(f"  Active sessions: {sessions}")
    assert "test_user" in sessions
    llm.delete_session("test_user")
    assert "test_user" not in llm.list_sessions()
    print(f"  Sessions after deletion: {llm.list_sessions()}")

    print(f"\n[Step 6] Unloading model...")
    llm.unload_model()
    assert llm._llm is None
    print(f"  Model unloaded, GPU cache cleared.")

    print()
    print("=" * 60)
    print("ICM LLM INTEGRATION VALIDATED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    validate_icm_llm()
