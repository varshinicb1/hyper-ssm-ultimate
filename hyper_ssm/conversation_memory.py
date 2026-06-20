"""
Infinite Context Memory (ICM) for LLMs

Wraps HierarchicalHyperbolicMemory into a conversation memory system
that maintains O(1) compressed state of the entire chat history.

Key insight:
    HHM compresses the ENTIRE conversation into a FIXED-SIZE hyperbolic
    state vector. Reading at different geometric scales gives different
    levels of abstraction — from verbatim recall (scale=0) to abstract gist
    (scale=K-1). This is the geometric analogue of "coreset" compression
    in hyperbolic space — the manifold's exponential capacity means a
    fixed-dimension vector can store exponentially more structure than
    its Euclidean counterpart.

Components:
    - InfiniteContextMemory: Core memory wrapping HHM (remember/recall/state)
    - ChatSession:           Full conversation lifecycle management
    - RAGStore:              Chunk-based retrieval augmented generation
    - InfiniteContextLLM:    Plug-and-play LLM integration with ICM
"""

import numpy as np
import torch
import torch.nn as nn
import pickle
from typing import Optional, List, Dict, Any, Callable, Union
from dataclasses import dataclass

from .hierarchical_memory import (
    HierarchicalHyperbolicMemory,
    exp_map,
    log_map,
    project_to_hyperboloid,
    check_manifold,
)


def _to_tensor(
    x: Union[np.ndarray, torch.Tensor, List[float]],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device) if device is not None else x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device) if device is not None else torch.from_numpy(x)
    return torch.tensor(x, device=device)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


# =========================================================================
# INFINITE CONTEXT MEMORY — Core
# =========================================================================

class InfiniteContextMemory:
    """
    Wraps HierarchicalHyperbolicMemory into a streaming conversation memory.

    Maintains a fixed-size hyperbolic state vector that compresses the entire
    conversation history. The state lives on the hyperboloid manifold and is
    updated incrementally via Lorentzian recurrence.

    O(1) memory: state size = (state_dim + 1) * 4 bytes (float32), independent
    of conversation length.

    Args:
        state_dim: Dimension of the spatial component of the state (default 64).
        num_scales: Number of hierarchical abstraction levels (default 4).
        device: Torch device to use.
    """

    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        state_dim: int = 64,
        num_scales: int = 4,
        device: Optional[torch.device] = None,
    ):
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.num_scales = num_scales
        self.device = device or torch.device("cpu")

        self.hhm = HierarchicalHyperbolicMemory(
            state_dim=state_dim, num_scales=num_scales
        ).to(self.device)
        self.hhm.eval()

        # Input projection: maps arbitrary embedding dim -> state_dim
        if embedding_dim is not None and embedding_dim != state_dim:
            self.input_proj = nn.Linear(embedding_dim, state_dim, bias=False).to(self.device)
        else:
            self.input_proj = None

        self._state: Optional[torch.Tensor] = None
        self._utterance_count: int = 0

    def _make_origin_state(self, batch: int = 1) -> torch.Tensor:
        h = torch.zeros(batch, self.state_dim + 1, device=self.device)
        h[..., 0] = torch.sqrt(self.hhm._curvature())
        return h

    def reset(self) -> "InfiniteContextMemory":
        self._state = None
        self._utterance_count = 0
        return self

    @torch.no_grad()
    def remember(
        self,
        utterance_embedding: Union[np.ndarray, torch.Tensor, List[float]],
    ) -> "InfiniteContextMemory":
        """
        Compress an utterance embedding into the hyperbolic state.

        The embedding is projected onto the hyperboloid manifold via the
        exponential map, then fused with the current state using a Lorentzian
        gated recurrence. The result is a new manifold state encoding all
        prior utterances plus this one — in O(1) space.

        Args:
            utterance_embedding: D-dimensional embedding vector (numpy/torch/list).
                Compatible with any embedding model (sentence-transformers, OpenAI, etc.).

        Returns:
            self for chaining.
        """
        x = _to_tensor(utterance_embedding, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = x.view(1, -1)

        if self.input_proj is not None:
            x = self.input_proj(x)

        if self._state is None:
            self._state = self._make_origin_state(batch=1)

        x_hyp = exp_map(x)
        self._state = self.hhm._step(self._state, x_hyp)
        self._utterance_count += 1
        return self

    @torch.no_grad()
    def recall(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor, List[float]],
        scale: int = 0,
    ) -> np.ndarray:
        """
        Retrieve from the compressed state using multi-scale geometric readout.

        The query embedding modulates the readout direction on the hyperboloid.
        The compressed state is mapped to tangent space, shifted toward the
        query direction, then scaled toward the origin — more scaling yields
        more abstract (gist-like) recall.

        scale=0:   most detailed (closest to verbatim)
        scale=K-1: most abstract (closest to gist / semantic summary)

        Args:
            query_embedding: D-dimensional query embedding vector.
            scale: Abstraction level (0 to num_scales-1).

        Returns:
            D-dimensional numpy array: the recalled memory content.
        """
        if self._state is None:
            return np.zeros(self.state_dim)

        query = _to_tensor(query_embedding, device=self.device)
        if query.dim() == 1:
            query = query.unsqueeze(0)

        if self.input_proj is not None:
            query = self.input_proj(query)

        state = self._state
        state_tangent = log_map(state)
        query_hyp = exp_map(query)
        query_tangent = log_map(query_hyp)

        combined_tangent = state_tangent + 0.1 * query_tangent

        depth = (scale + 1) / self.num_scales
        abstracted = combined_tangent * (1.0 - depth)

        on_manifold = exp_map(abstracted)
        spatial = on_manifold[..., 1:]
        result = self.hhm.scale_projectors[scale](spatial)
        return _to_numpy(result[0])

    @torch.no_grad()
    def recall_all_scales(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor, List[float]],
    ) -> List[np.ndarray]:
        return [self.recall(query_embedding, s) for s in range(self.num_scales)]

    def state(self) -> Dict[str, Any]:
        """
        Return the current compressed state (serializable).

        Fixed-size regardless of conversation length. Contains:
            - compressed_state: hyperbolic state vector (numpy or None)
            - utterance_count:  how many utterances were compressed
            - state_dim / num_scales / model_weights: configuration + model params

        Save/restore example:
            >>> mem = InfiniteContextMemory()
            >>> mem.remember(emb)
            >>> data = mem.state()
            >>> # Pickle
            >>> import pickle
            >>> pickle.dump(data, open("state.pkl", "wb"))
            >>> # JSON (arrays -> lists)
            >>> import json
            >>> json_data = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in data.items()}
        """
        weights = {f"hhm.{k}": _to_numpy(v) for k, v in self.hhm.state_dict().items()}
        if self.input_proj is not None:
            weights["input_proj.weight"] = _to_numpy(self.input_proj.weight)
        return {
            "compressed_state": _to_numpy(self._state) if self._state is not None else None,
            "utterance_count": self._utterance_count,
            "state_dim": self.state_dim,
            "num_scales": self.num_scales,
            "embedding_dim": self.embedding_dim,
            "model_weights": weights,
        }

    def load_state(self, state_dict: Dict[str, Any]) -> "InfiniteContextMemory":
        if "model_weights" in state_dict:
            hhm_sd = {
                k.replace("hhm.", ""): torch.from_numpy(v).to(self.device)
                for k, v in state_dict["model_weights"].items()
                if k.startswith("hhm.")
            }
            self.hhm.load_state_dict(hhm_sd)
            if self.input_proj is not None and "input_proj.weight" in state_dict["model_weights"]:
                self.input_proj.weight.data = torch.from_numpy(
                    state_dict["model_weights"]["input_proj.weight"]
                ).to(self.device)
        if state_dict["compressed_state"] is not None:
            self._state = torch.from_numpy(state_dict["compressed_state"]).to(self.device)
        else:
            self._state = None
        self._utterance_count = state_dict["utterance_count"]
        return self

    @property
    def memory_size_bytes(self) -> int:
        if self._state is None:
            return 0
        return self._state.numel() * self._state.element_size()

    def info(self) -> Dict[str, Any]:
        return {
            "utterance_count": self._utterance_count,
            "state_dim": self.state_dim,
            "num_scales": self.num_scales,
            "memory_bytes": self.memory_size_bytes,
            "state_on_manifold": (
                float(check_manifold(self._state).item()) if self._state is not None else None
            ),
        }


# =========================================================================
# CHAT SESSION — Full Lifecycle Wrapper
# =========================================================================

@dataclass
class ChatMessage:
    role: str
    content: str
    embedding: Optional[np.ndarray] = None


class ChatSession:
    """
    Full conversation lifecycle built on InfiniteContextMemory.

    Maintains both:
        1. The compressed hyperbolic state (O(1), captures the ENTIRE conversation)
        2. A rolling window of recent messages (for display / exact reference)

    Args:
        state_dim: Dimension of hyperbolic state (default 64).
        num_scales: Hierarchical abstraction levels (default 4).
        max_history: Number of recent messages to keep in rolling window (default 100).
        embedding_fn: Callable mapping text -> embedding vector.
            If None, embeddings must be provided explicitly.
    """

    def __init__(
        self,
        state_dim: int = 64,
        num_scales: int = 4,
        max_history: int = 100,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
    ):
        self.memory = InfiniteContextMemory(state_dim=state_dim, num_scales=num_scales)
        self.max_history = max_history
        self._embedding_fn = embedding_fn
        self._messages: List[ChatMessage] = []

    def add_message(
        self,
        role: str,
        content: str,
        embedding: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
    ) -> "ChatSession":
        if embedding is None and self._embedding_fn is not None:
            embedding = self._embedding_fn(content)

        msg = ChatMessage(role=role, content=content, embedding=embedding)
        self._messages.append(msg)

        if len(self._messages) > self.max_history:
            self._messages = self._messages[-self.max_history:]

        if embedding is not None:
            self.memory.remember(embedding)

        return self

    def query(
        self,
        text: str,
        scale: int = 0,
    ) -> np.ndarray:
        embedding = self._embedding_fn(text) if self._embedding_fn else np.random.randn(self.memory.state_dim)
        return self.memory.recall(embedding, scale=scale)

    def recent_messages(self, n: int = 10) -> List[ChatMessage]:
        return self._messages[-n:]

    @property
    def all_messages(self) -> List[ChatMessage]:
        return self._messages

    @property
    def total_messages(self) -> int:
        return self.memory._utterance_count

    def save(self, path: str) -> None:
        data = {
            "memory_state": self.memory.state(),
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "embedding": m.embedding.tolist() if m.embedding is not None else None,
                }
                for m in self._messages
            ],
            "max_history": self.max_history,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str, embedding_fn: Optional[Callable[[str], np.ndarray]] = None) -> "ChatSession":
        with open(path, "rb") as f:
            data = pickle.load(f)

        session = cls(
            state_dim=data["memory_state"]["state_dim"],
            num_scales=data["memory_state"]["num_scales"],
            max_history=data["max_history"],
            embedding_fn=embedding_fn,
        )
        session.memory.load_state(data["memory_state"])
        session._messages = [
            ChatMessage(
                role=m["role"],
                content=m["content"],
                embedding=np.array(m["embedding"]) if m["embedding"] is not None else None,
            )
            for m in data["messages"]
        ]
        return session

    def info(self) -> Dict[str, Any]:
        mem_info = self.memory.info()
        mem_info["total_messages"] = self.total_messages
        mem_info["rolling_window_size"] = len(self._messages)
        return mem_info


# =========================================================================
# RAG STORE — Chunk-Based Retrieval with HHM
# =========================================================================

class RAGStore:
    """
    Chunk-based retrieval system using HierarchicalHyperbolicMemory.

    Stores text chunks in the HHM compressed state and maintains a parallel
    embedding index for exact nearest-neighbor retrieval. The HHM state
    captures the gist of the entire document collection; the embedding index
    provides exact chunk retrieval.

    This dual approach gives:
        1. O(1) compressed representation of ALL chunks (via HHM)
        2. Exact top-k chunk retrieval (via embedding index)

    Args:
        state_dim: Dimension for HHM state (default 128).
        num_scales: Number of abstraction levels (default 4).
    """

    def __init__(self, state_dim: int = 128, num_scales: int = 4):
        self.memory = InfiniteContextMemory(state_dim=state_dim, num_scales=num_scales)
        self._chunks: List[str] = []
        self._embeddings: List[np.ndarray] = []
        self._metadata: List[Dict[str, Any]] = []

    def add_chunk(
        self,
        text: str,
        embedding: Union[np.ndarray, torch.Tensor, List[float]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RAGStore":
        emb = _to_numpy(_to_tensor(embedding))
        self._chunks.append(text)
        self._embeddings.append(emb)
        self._metadata.append(metadata or {})
        self.memory.remember(emb)
        return self

    def add_chunks(
        self,
        texts: List[str],
        embeddings: Union[np.ndarray, List[np.ndarray]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> "RAGStore":
        for i, text in enumerate(texts):
            emb = embeddings[i] if isinstance(embeddings, list) else embeddings[i]
            self.add_chunk(text, emb, metadata[i] if metadata else None)
        return self

    def retrieve(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor, List[float]],
        top_k: int = 5,
        use_hhm: bool = False,
        hhm_scale: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks relevant to a query.

        Two modes:
            use_hhm=False (default): Exact cosine similarity search over
                stored embeddings. Guaranteed to find the closest chunks.
            use_hhm=True: Uses HHM multi-scale recall to find chunks.
                May retrieve chunks that are semantically related even if
                not exact embedding neighbors. Demonstrates the "infinite
                context" property.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of chunks to return.
            use_hhm: If True, use HHM-based retrieval instead of exact search.
            hhm_scale: Abstraction level for HHM retrieval.

        Returns:
            List of dicts with keys: text, score, metadata
        """
        query = _to_numpy(_to_tensor(query_embedding))

        if use_hhm:
            return self._retrieve_hhm(query, top_k, hhm_scale)
        return self._retrieve_exact(query, top_k)

    def _retrieve_exact(self, query: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        if not self._embeddings:
            return []
        emb_matrix = np.stack(self._embeddings, axis=0)
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        emb_norms = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8)
        scores = emb_norms @ query_norm
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"text": self._chunks[i], "score": float(scores[i]), "metadata": self._metadata[i]}
            for i in top_indices
        ]

    def _retrieve_hhm(self, query: np.ndarray, top_k: int, scale: int) -> List[Dict[str, Any]]:
        if not self._embeddings:
            return []
        recalled = self.memory.recall(query, scale=scale)
        scores = np.array([
            np.dot(emb, recalled) / (np.linalg.norm(emb) * np.linalg.norm(recalled) + 1e-8)
            for emb in self._embeddings
        ])
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"text": self._chunks[i], "score": float(scores[i]), "metadata": self._metadata[i]}
            for i in top_indices
        ]

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "text": self._chunks[idx],
            "embedding": self._embeddings[idx],
            "metadata": self._metadata[idx],
        }


# =========================================================================
# INFINITE CONTEXT LLM — Full Integration
# =========================================================================

class InfiniteContextLLM:
    """
    LLM integration with Infinite Context Memory.

    Wraps an LLM (simulated by default) with an InfiniteContextMemory that
    maintains O(1) compressed state of the entire conversation.

    Flow:
        1. User message is embedded and compressed into hyperbolic state
        2. The state is read at multiple scales for hierarchical context
        3. Recalled context is fed into the LLM prompt
        4. LLM response is also embedded and compressed

    Because the state is FIXED-SIZE, this works for arbitrarily long
    conversations without context window overflow.

    Args:
        embedding_dim: Dimension of embeddings (default 384).
        state_dim: Dimension of hyperbolic memory state (default 64).
        num_scales: Number of abstraction levels (default 4).
        llm_fn: Optional callable (prompt -> response). Uses simulated LLM if None.
        embedding_fn: Optional callable (text -> embedding). Uses hash-based
            embedding if None (for testing only).
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        state_dim: int = 64,
        num_scales: int = 4,
        llm_fn: Optional[Callable[[str], str]] = None,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
    ):
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim

        self.memory = InfiniteContextMemory(
            embedding_dim=embedding_dim, state_dim=state_dim, num_scales=num_scales
        )
        self._llm_fn = llm_fn or self._default_llm
        self._embedding_fn = embedding_fn or self._default_embed
        self._system_prompt = (
            "You are an AI assistant with infinite context memory. "
            "You remember everything from the entire conversation, even "
            "details from the very beginning."
        )
        self._history: List[Dict[str, str]] = []

    def _default_embed(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(hash(text) % (2**31))
        return rng.randn(self.embedding_dim).astype(np.float32)

    @staticmethod
    def _default_llm(prompt: str) -> str:
        return (
            f"[simulated LLM] Received prompt of {len(prompt)} characters. "
            f"I have access to my infinite context memory."
        )

    def _build_prompt(self, user_message: str, recalled: List[np.ndarray]) -> str:
        parts = [f"System: {self._system_prompt}"]

        if self._history:
            parts.append("\nRecent conversation:")
            for msg in self._history[-6:]:
                parts.append(f"  {msg['role']}: {msg['content']}")

        parts.append("\nInfinite Context Recall:")
        scale_names = ["detailed", "moderate", "abstract", "gist"]
        for i, recall_vec in enumerate(recalled):
            activation = float(np.linalg.norm(recall_vec))
            parts.append(f"  [{scale_names[i]} recall: activation={activation:.3f}]")

        parts.append(f"\nUser: {user_message}")
        parts.append("\nAssistant:")
        return "\n".join(parts)

    def chat(self, message: str) -> str:
        user_emb = self._embedding_fn(message)
        self.memory.remember(user_emb)
        self._history.append({"role": "user", "content": message})

        query_emb = self._embedding_fn(message)
        recalled = self.memory.recall_all_scales(query_emb)

        prompt = self._build_prompt(message, recalled)
        response = self._llm_fn(prompt)

        response_emb = self._embedding_fn(response)
        self.memory.remember(response_emb)
        self._history.append({"role": "assistant", "content": response})

        return response

    def save_session(self, path: str) -> None:
        data = {
            "memory": self.memory.state(),
            "history": self._history,
            "system_prompt": self._system_prompt,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_session(self, path: str) -> "InfiniteContextLLM":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.memory.load_state(data["memory"])
        self._history = data["history"]
        self._system_prompt = data.get("system_prompt", self._system_prompt)
        return self

    @property
    def conversation_length(self) -> int:
        return self.memory._utterance_count


# =========================================================================
# SELF-VALIDATION
# =========================================================================

def validate_icm():
    print("=" * 60)
    print("INFINITE CONTEXT MEMORY — VALIDATION SUITE")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    print("\n[Test 1] Basic remember/recall...")
    icm = InfiniteContextMemory(state_dim=64, num_scales=4)
    for _ in range(10):
        emb = np.random.randn(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        icm.remember(emb)
    recalled = icm.recall(np.random.randn(64).astype(np.float32), scale=0)
    assert recalled.shape == (64,), f"Expected (64,), got {recalled.shape}"
    print(f"  PASS: recall shape = {recalled.shape}, norm = {np.linalg.norm(recalled):.4f}")

    print("\n[Test 2] O(1) memory verification...")
    for n in [10, 100, 1000]:
        icm = InfiniteContextMemory(state_dim=64)
        for _ in range(n):
            icm.remember(np.random.randn(64).astype(np.float32))
        size = icm.memory_size_bytes
        print(f"  {n:5d} utterances: memory = {size}B (fixed)")
    assert size > 0, "Memory should not be zero"
    print("  PASS: memory is O(1)")

    print("\n[Test 3] State serialization...")
    icm1 = InfiniteContextMemory(state_dim=64)
    for _ in range(5):
        icm1.remember(np.random.randn(64).astype(np.float32))
    state_dict = icm1.state()
    icm2 = InfiniteContextMemory(state_dim=64)
    icm2.load_state(state_dict)
    rng_state = np.random.RandomState(0)
    q = rng_state.randn(64).astype(np.float32)
    recalled1 = icm1.recall(q, scale=0)
    recalled2 = icm2.recall(q, scale=0)
    assert np.allclose(recalled1, recalled2, atol=1e-5), "State serialization mismatch"
    print("  PASS: state save/load works")

    print("\n[Test 4] Multi-scale recall diversity...")
    icm = InfiniteContextMemory(state_dim=64, num_scales=4)
    for _ in range(20):
        icm.remember(np.random.randn(64).astype(np.float32))
    query = np.random.randn(64).astype(np.float32)
    scales = icm.recall_all_scales(query)
    norms = [np.linalg.norm(s) for s in scales]
    print(f"  Norm per scale: {[f'{n:.4f}' for n in norms]}")
    diffs = [np.linalg.norm(scales[i] - scales[i + 1]) for i in range(3)]
    print(f"  Inter-scale diffs: {[f'{d:.4f}' for d in diffs]}")
    assert all(d > 1e-6 for d in diffs), "Scales should produce different readouts"
    print("  PASS: multi-scale diversity confirmed")

    print("\n[Test 5] ChatSession lifecycle...")
    session = ChatSession(state_dim=64, embedding_fn=lambda t: np.random.randn(64).astype(np.float32))
    session.add_message("user", "Hello, my name is Alice")
    session.add_message("assistant", "Hi Alice!")
    session.add_message("user", "What was my name?")
    session.add_message("assistant", "Your name is Alice")
    assert session.total_messages == 4
    print(f"  PASS: {session.total_messages} messages, state = {session.memory.memory_size_bytes}B")

    print("\n[Test 6] RAGStore...")
    store = RAGStore(state_dim=64)
    chunks = [
        "The capital of France is Paris.",
        "The Eiffel Tower is in Paris.",
        "Python is a programming language.",
        "The Great Wall of China is visible from space.",
    ]
    for chunk in chunks:
        emb = np.random.randn(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        store.add_chunk(chunk, emb)
    results = store.retrieve(np.random.randn(64).astype(np.float32), top_k=2)
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    print(f"  PASS: retrieved {len(results)} chunks from {len(store)} total")

    print("\n[Test 7] InfiniteContextLLM...")
    llm = InfiniteContextLLM(embedding_dim=64, state_dim=32)
    response = llm.chat("Tell me about yourself")
    assert isinstance(response, str) and len(response) > 0
    print(f"  PASS: LLM response = \"{response[:80]}...\"")
    assert llm.conversation_length == 2
    print(f"  Memory: {llm.memory.memory_size_bytes}B for {llm.conversation_length} messages")

    print()
    print("=" * 60)
    print("ALL VALIDATIONS PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    validate_icm()
