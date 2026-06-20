# Promotion Plan — Infinite Context Memory (ICM)

## 1. Upload Social Preview

Go to: https://github.com/varshinicb1/hyper-ssm-ultimate/settings
Scroll to "Social preview" → Upload `social-preview.png`
→ This makes the repo card show up on Twitter/HN/Reddit/LinkedIn

## 2. Hacker News Post

URL: https://news.ycombinator.com/submit

**Title:** 260 bytes per conversation forever — O(1) hyperbolic memory for LLMs

**URL:** https://github.com/varshinicb1/hyper-ssm-ultimate

**Optional text:**
Replace the KV-cache with a fixed-size hyperbolic state vector. Works with any HuggingFace model. 68 tests, pip install, Docker one-liner, WebSocket streaming, 4-bit quantization.

## 3. Reddit Posts

### r/MachineLearning
https://www.reddit.com/r/MachineLearning/submit

**Title:** [P] Infinite Context Memory (ICM) — O(1) hyperbolic memory for any LLM, 260 bytes per session

**Text:**
ICM replaces the KV-cache with a fixed-size hyperbolic state vector. The state is 260 bytes — regardless of whether the conversation is 10 turns or 10 million.

Key properties:
- O(1) memory: fixed 260B state, never grows
- Multi-scale readout at 4 abstraction levels (detail → gist)
- Works with any HuggingFace causal LM (GPT-2, Qwen, Phi-3, etc.)
- Quantization: 4-bit NF4 and 8-bit via bitsandbytes
- Persistent sessions via SQLite
- Token-by-token streaming (WebSocket + SSE)
- Docker one-command deploy
- 68 passing tests, zero warnings

pip install icm-llm && icm-demo

GitHub: https://github.com/varshinicb1/hyper-ssm-ultimate

### r/LocalLLaMA
https://www.reddit.com/r/LocalLLaMA/submit

**Title:** ICM: O(1) memory for local LLMs — 260 bytes per session, works with any HF model

**Text:**
Built this for my RTX 4050 laptop. 260 bytes of state per conversation, regardless of length. Supports 4-bit quantization, runs with GPT-2, Qwen, Phi-3, TinyLlama. Web UI, REST API, Docker.

pip install icm-llm

## 4. Twitter/LinkedIn

**Text:**
260 bytes per conversation. Forever.

Infinite Context Memory (ICM) replaces the KV-cache with a fixed-size hyperbolic state vector for any LLM.

O(1) memory. Open source. pip install icm-llm

https://github.com/varshinicb1/hyper-ssm-ultimate

## 5. Awesome Lists

Submit to:
- https://github.com/curated/awesome-llm (PR to README.md)
- https://github.com/Hannibal046/Awesome-LLM (PR to README.md)
- https://github.com/kyrolabs/awesome-langchain (if relevant)

## 6. Colab Notebook

Share: https://colab.research.google.com/github/varshinicb1/hyper-ssm-ultimate/blob/main/icm_demo.ipynb
