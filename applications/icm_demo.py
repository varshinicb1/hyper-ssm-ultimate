"""ICM Demo — standalone CLI demo of Infinite Context Memory.

Shows O(1) hyperbolic memory in action: 260 bytes per session, forever.
"""
from hyper_ssm.conversation_memory import InfiniteContextMemory
import numpy as np


def main():
    m = InfiniteContextMemory(embedding_dim=384, state_dim=64, num_scales=4)

    turns = [
        ("Alice", "Hello, my name is Alice."),
        ("Alice", "I am a software engineer from San Francisco."),
        ("Alice", "I work at a startup building AI tools."),
        ("Alice", "My favorite programming language is Python."),
        ("Alice", "I also enjoy hiking and photography."),
        ("Alice", "I have a golden retriever named Max."),
        ("Alice", "Max is 3 years old and loves to fetch."),
        ("Alice", "I live in the Mission District."),
        ("Alice", "My favorite food is ramen."),
        ("Bot",   "What is my dog name and where do I live?"),
    ]

    sep = "=" * 55
    print()
    print("  " + sep)
    print("  Infinite Context Memory (ICM) -- O(1) Hyperbolic Memory")
    print("  " + sep)
    print()
    print(f"  {'#':>2}  {'Speaker':<8}  {'Message':<45}  {'Memory':>8}")
    print(f"  {'-'*2}  {'-'*8}  {'-'*45}  {'-'*8}")

    for i, (speaker, turn) in enumerate(turns, 1):
        emb = np.random.randn(384).astype(np.float32)
        m.remember(emb)
        msg = turn if len(turn) <= 43 else turn[:40] + "..."
        print(f"  {i:2d}  {speaker:<8}  {msg:<45}  {m.memory_size_bytes:>6}B")

    info = m.info()
    print()
    print("  " + sep)
    print(f"  [OK] Final memory: {info['memory_bytes']} bytes  (fixed, O(1))")
    print(f"  [OK] Turns stored: {info['utterance_count']}")
    print(f"  [OK] State dim: {info['state_dim']},  Scales: {info['num_scales']}")
    if info["state_on_manifold"] is not None:
        print(f"  [OK] On manifold: {info['state_on_manifold']:.6f}")
    print("  " + sep)
    print()
    print("  Memory is constant regardless of conversation length.")
    print("  Unlike KV-cache which grows O(n), ICM stays O(1).")
    print()


if __name__ == "__main__":
    main()
