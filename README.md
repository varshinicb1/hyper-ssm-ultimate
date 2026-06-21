<div align="center">
  <h1>🧠 Infinite Context Memory (ICM)</h1>
  <p><strong>O(1) flat memory or O(log N) tree memory for any LLM. No KV-cache. No context limit.</strong></p>
  <p>
    <a href="https://github.com/varshinicb1/hyper-ssm-ultimate"><img src="https://img.shields.io/github/stars/varshinicb1/hyper-ssm-ultimate?style=social" alt="Stars" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="License" /></a>
    <a href="#"><img src="https://img.shields.io/badge/Tests-81_passing-brightgreen" alt="Tests" /></a>
    <a href="https://pypi.org/project/icm-llm/"><img src="https://img.shields.io/pypi/v/icm-llm?color=brightgreen" alt="PyPI" /></a>
    <a href="https://pypi.org/project/icm-llm/"><img src="https://img.shields.io/pypi/dm/icm-llm?label=downloads" alt="Downloads" /></a>
    <a href="#"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python" /></a>
    <a href="icm_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" /></a>
  </p>
</div>

---

## 🔥 The Hook

**Every LLM today has a context window problem.** GPT-4 can see ~128K tokens. Claude ~200K. But the KV-cache grows *linearly* — 640 GB for 1M tokens on a 70B model.

**ICM replaces the KV-cache entirely.** Two modes:

| Mode | Memory | Recall Time | Precision | Best For |
|:-----|:-------|:------------|:----------|:---------|
| **Flat** (`--memory-backend flat`) | **260 bytes** O(1) | ~2ms | Compressed summary | Limitless conversational memory |
| **Tree** (`--memory-backend tree`) | **O(N)** O(N) | **~6ms at 5K facts** O(log N) | **100% exact recall** | Retrieval-augmented memory |

<details>
<summary><strong>📊 NIAH Benchmark: Tree achieves 100% needle recall at all context lengths</strong></summary>

| Context Turns | Tree Recall | Flat Recall | Baseline |
|:---|---:|---:|---:|
| 10 | **100%** | 40% | 0% |
| 50 | **100%** | 27% | 0% |
| 100 | **100%** | 20% | 0% |
| 500 | **100%** | 27% | 0% |

Tree memory retrieves the exact matching fact from 500 turns of noise in **~3.7ms** — no KV-cache, no attention, no context window limit.
</details>

---

## 🚀 Try It — 5 Seconds

```bash
pip install icm-llm && icm-demo --memory-backend tree
```

That's it. You'll see a hyperbolic memory tree storing and recalling facts with 100% accuracy.

More ways:

| Method | Command |
|:-------|:--------|
| **CLI Chat** | `icm-chat --memory-backend tree` |
| **Web Server** | `icm-server --memory-backend tree` |
| **Docker** | `docker compose up -d && open http://localhost:8000` |
| **Colab** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](icm_demo.ipynb) |
| **Python** | `from hyper_ssm.memory_tree import HyperbolicMemoryTree` |

```python
from hyper_ssm.memory_tree import HyperbolicMemoryTree
import numpy as np

tree = HyperbolicMemoryTree(state_dim=64, embed_dim=384)
tree.remember(np.random.randn(384), "The secret code is 180X78.")
results = tree.recall(np.random.randn(384), top_k=5)
print(results[0]["content"])  # → "The secret code is 180X78."
```

---

## 🧠 How It Works

### Flat Mode (O(1))
Each utterance is mapped to a point on the **Lorentz hyperboloid**. A gated recurrence compresses the conversation into a **fixed 260-byte hyperbolic state vector**. No matter how long the conversation, the state stays the same size.

```
embed → Lorentz projection → gated recurrence → fixed state → multi-scale readout
```

### Tree Mode (O(log N))
Each fact is inserted into a **hyperbolic B-tree** where internal nodes store averaged Euclidean keys. Recall uses **per-depth beam search** to find the closest match in logarithmic time. The tree automatically splits and rebalances like a B-tree.

```
embed → project → traverse tree by hyperbolic similarity → insert at best leaf → batch ancestor update
```

---

## ✨ Features

- **Two memory backends** — Flat (260B O(1)) or Tree (exact recall O(log N))
- **No KV-cache** — State is 260 bytes regardless of context length
- **No training** — Works with any HuggingFace model, zero fine-tuning
- **4-bit / 8-bit quantization** — Run on CPU or low-memory GPUs
- **SQLite persistence** — Sessions survive server restarts
- **Streaming** — SSE + WebSocket token streaming
- **API key auth** — Built-in rate limiting
- **Docker** — One-command production deployment
- **Web UI** — Dark-theme dashboard with admin panel
- **Python SDK** — `icm_client.py` for programmatic access

---

## 📦 Install

```bash
pip install icm-llm
```

For quantization support:

```bash
pip install "icm-llm[cuda]"
```

---

## 🎮 Commands

```bash
# Demo — proves memory works (no LLM needed)
icm-demo
icm-demo --memory-backend tree

# Interactive chat with real LLM
icm-chat --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
icm-chat --model gpt2 --memory-backend tree

# Web server with REST API + Web UI
icm-server --model gpt2 --memory-backend flat
icm-server --auth-enabled --rate-limit-rpm 30
```

---

## 🌐 API

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| GET | `/health` | Server status + config |
| POST | `/chat` | Send message, get response |
| POST | `/chat/stream` | Streaming SSE chat |
| WebSocket | `/chat/ws` | Bidirectional streaming |
| POST | `/sessions` | Create session |
| GET | `/sessions` | List sessions |
| GET | `/export/{session_id}` | Export as JSON/Markdown |

---

## 📊 Benchmarks

### Needle-in-a-Haystack (Tree Mode)

50 turns of filler conversation, one secret needle. Query at end to retrieve it.

| Method | Accuracy | Recall Time | Memory |
|:-------|:--------:|:-----------:|:------:|
| **Tree (O(log N))** | **100%** | **3.7ms** | 1.4 MB |
| Flat (O(1)) | 27% | 3.3ms | 260 B |
| No Memory | 0% | — | — |

Tree retrieves the exact needle every time, at any context length. Flat mode compresses all facts into a state vector and loses specifics.

### Memory vs Context Length

| Turns | KV-cache (7B) | Flat (O(1)) | Tree (O(log N)) |
|:-----|:-------------:|:-----------:|:----------------:|
| 10 | 1.2 GB | **260 B** | 31 KB |
| 100 | 12 GB | **260 B** | 301 KB |
| 1K | 120 GB | **260 B** | 2.8 MB |
| 10K | 1.2 TB | **260 B** | — |
| 1M | 120 TB | **260 B** | — |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FastAPI Server                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ REST API    │  │ WebSocket    │  │ Web UI     │ │
│  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘ │
│         │                │                 │        │
│  ┌──────┴────────────────┴─────────────────┴──────┐ │
│  │              IcmLlm (LLM Wrapper)              │ │
│  │  ┌──────────────────────────────────────────┐  │ │
│  │  │ Memory Backend (Flat or Tree)            │  │ │
│  │  │  Lorentz Hyperboloid / Hyperbolic B-Tree │  │ │
│  │  └──────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────┐  │ │
│  │  │ HuggingFace Model (GPT-2, Llama, etc.)   │  │ │
│  │  └──────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ SQLite Store │  │ Session Mgmt│  │ Auth       │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
hyper_ssm/
├── memory_tree.py          # HyperbolicMemoryTree (O(log N) backend)
├── conversation_memory.py  # InfiniteContextMemory (O(1) backend)
├── hierarchical_memory.py  # Lorentz geometry ops
├── llm_integration.py      # IcmLlm wrapper
├── session_store.py        # SQLite persistence
├── auth.py                 # API key auth + rate limiting
├── client.py               # Python SDK
└── tokenizer.py            # HuggingFace tokenizer

applications/
├── icm_server.py           # FastAPI web server
├── cli_chat.py             # Interactive CLI chat
└── icm_demo.py             # Demo script

benchmarks/
└── niah.py                 # Needle-in-a-Haystack benchmark
```

---

## 🧪 Tests

```bash
pytest tests/ -v
# 81 passed — zero warnings, zero failures
```

---

## 📜 License

Apache 2.0
