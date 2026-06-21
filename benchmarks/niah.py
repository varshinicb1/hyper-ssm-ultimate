"""
Needle-in-a-Haystack benchmark: Hyperbolic Memory Tree vs Flat ICM vs No Memory.

Measures recall accuracy, retrieval latency, and memory usage at
increasing context lengths (10 to 1000 turns).
"""

import time
import gc
import os
import sys
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

FILLER_TEMPLATES = [
    "The weather today is {adj} with a high of {temp}C.",
    "I read an interesting article about {topic} in the newspaper.",
    "My friend told me about their trip to {place} last weekend.",
    "The new restaurant on {street} serves amazing {food}.",
    "I just finished reading {book} by {author}.",
    "Today I learned how to {skill} in just a few minutes.",
    "The movie {movie} was {adj2} than I expected.",
    "I need to buy {item} from the grocery store later.",
    "My favorite song right now is {song} by {artist}.",
    "The meeting at {time} was {adv} productive.",
]

FILLER_VOCAB = {
    "adj": ["sunny", "cloudy", "windy", "humid", "crisp", "warm", "mild"],
    "temp": [str(t) for t in range(15, 38)],
    "topic": ["quantum physics", "ancient Rome", "AI ethics", "oceanography",
              "renaissance art", "cryptography", "urban farming"],
    "place": ["Tokyo", "Barcelona", "Reykjavik", "Kathmandu", "Melbourne",
              "Cairo", "Stockholm"],
    "street": ["Main St", "Oak Ave", "Park Blvd", "Elm St", "Broadway", "River Rd"],
    "food": ["sushi", "tacos", "pasta", "curry", "ramen", "pizza", "dumplings"],
    "book": ["Dune", "Neuromancer", "Snow Crash", "Hyperion", "Anathem",
             "The Left Hand of Darkness", "Foundation"],
    "author": ["Frank Herbert", "William Gibson", "Neal Stephenson",
               "Dan Simmons", "Ursula K. Le Guin", "Isaac Asimov"],
    "skill": ["whittle wood", "bake sourdough", "code in Rust", "play chess",
              "meditate", "solve a Rubik cube"],
    "movie": ["Inception", "The Matrix", "Arrival", "Blade Runner 2049",
              "Interstellar", "Ex Machina"],
    "adj2": ["better", "worse", "more engaging", "less confusing"],
    "item": ["milk", "eggs", "bread", "avocados", "tofu", "quinoa", "kombucha"],
    "song": ["Bohemian Rhapsody", "Stairway to Heaven", "Hotel California",
             "Imagine", "Smells Like Teen Spirit"],
    "artist": ["Queen", "Led Zeppelin", "Eagles", "John Lennon", "Nirvana"],
    "time": ["10 AM", "2 PM", "3:30 PM", "9 AM", "1 PM", "4:15 PM"],
    "adv": ["surprisingly", "remarkably", "unusually", "quite", "very"],
}


def _fill(template: str, rng: np.random.Generator) -> str:
    import random
    result = template
    for key, values in FILLER_VOCAB.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, str(rng.choice(values)), 1)
    return result


def generate_filler_turn(turn_idx: int, rng: np.random.Generator) -> Tuple[str, str]:
    template = FILLER_TEMPLATES[turn_idx % len(FILLER_TEMPLATES)]
    topic = FILLER_TOPICS[turn_idx % len(FILLER_TOPICS)]
    return _fill(template, rng), topic


def generate_needle(needle_id: int, rng: np.random.Generator) -> Tuple[str, str, str]:
    code = f"{rng.integers(100, 999)}X{rng.integers(10, 99)}"
    word = rng.choice(["Zephyr", "Nebula", "Aether", "Quantum", "Phoenix",
                        "Cascade", "Horizon", "Tempest", "Aurora", "Onyx"])
    num = rng.integers(1000, 9999)
    key = f"{rng.integers(100000, 999999):06d}"
    templates = [
        f"The secret code is {code}. Remember it: {code}.",
        f"The password for the vault is {word}-{num}.",
        f"The access key is {key}. Keep it safe.",
    ]
    content = templates[needle_id % len(templates)]
    query = "What is the secret code or password or key?"
    return query, content, "secret"


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Controlled clustered embeddings
# ---------------------------------------------------------------------------

TOPIC_ANCHORS = {}
_TOPIC_RNG = np.random.default_rng(2024)

def _get_anchor(topic: str) -> np.ndarray:
    if topic not in TOPIC_ANCHORS:
        v = _TOPIC_RNG.standard_normal(384).astype(np.float32)
        TOPIC_ANCHORS[topic] = v / np.linalg.norm(v)
    return TOPIC_ANCHORS[topic]

# Topics cycle through these for filler turns
FILLER_TOPICS = ["weather", "reading", "travel", "food", "tech", "music", "sports"]

def embed(text: str, topic: Optional[str] = None) -> np.ndarray:
    """Deterministic embedding with controlled clustering.
    
    Facts in the same topic have nearby embeddings (anchor + small noise),
    mimicking real semantic embeddings. This allows the tree's hyperbolic
    similarity routing to actually work.
    
    Args:
        text: The text to embed (used for seed within topic).
        topic: Topic cluster. If None, inferred from text content or random.
    """
    if topic is None:
        # Infer topic from text
        for t in FILLER_TOPICS + ["secret", "Al_key"]:
            if t in text.lower() or t.replace("_", " ") in text.lower():
                topic = t
                break
        if topic is None:
            topic = "default"
    
    anchor = _get_anchor(topic)
    h = hash(text) & 0xFFFFFFFF
    rng = np.random.default_rng(h)
    noise = rng.standard_normal(384).astype(np.float32) * 0.08  # small noise
    noise = noise / np.linalg.norm(noise) * 0.08
    v = anchor + noise
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Eval types
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    backend: str
    context_turns: int
    needle_position_pct: float
    needle_retrieved: bool
    retrieval_time_ms: float
    memory_bytes: int
    nodes: Optional[int] = None
    depth: Optional[int] = None


# ---------------------------------------------------------------------------
# Evaluate one backend
# ---------------------------------------------------------------------------

def evaluate_backend(
    backend_name: str,
    num_turns_list: List[int] = (10, 50, 100, 500),
    needles_per_config: int = 5,
    positions: List[float] = (0.25, 0.50, 0.75),
) -> List[EvalResult]:
    results = []
    embed_dim = 384

    for num_turns in num_turns_list:
        for ni in range(needles_per_config):
            for pos_pct in positions:
                seed = hash(f"{backend_name}_{num_turns}_{ni}_{pos_pct}") & 0xFFFFFFFF
                rng = np.random.default_rng(seed)

                # Create fresh backend
                if backend_name == "flat":
                    from hyper_ssm.conversation_memory import InfiniteContextMemory
                    mem = InfiniteContextMemory(
                        embedding_dim=embed_dim, state_dim=64, num_scales=4,
                        device=None,
                    )
                elif backend_name == "tree":
                    from hyper_ssm.memory_tree import HyperbolicMemoryTree
                    mem = HyperbolicMemoryTree(state_dim=64, embed_dim=embed_dim)
                else:
                    mem = None

                # Generate needle
                _, needle_content, needle_topic = generate_needle(ni, rng)
                needle_emb = embed(needle_content, needle_topic)

                # Build context with needle at target position
                needle_turn = max(0, min(num_turns - 1, int(num_turns * pos_pct)))
                for i in range(num_turns):
                    if i == needle_turn:
                        if backend_name == "tree" and mem is not None:
                            mem.remember(needle_emb, needle_content)
                        elif mem is not None:
                            mem.remember(needle_emb)
                    else:
                        filler, filler_topic = generate_filler_turn(i, rng)
                        filler_emb = embed(filler, filler_topic)
                        if backend_name == "tree" and mem is not None:
                            mem.remember(filler_emb, filler)
                        elif mem is not None:
                            mem.remember(filler_emb)

                # Query + measure retrieval
                query_emb = embed("What is the secret code or password or key?")
                t0 = time.perf_counter()
                retrieved = False

                if backend_name == "flat":
                    recalled = mem.recall_all_scales(query_emb)
                    if len(recalled) > 0 and all(v is not None and v.size > 0 for v in recalled):
                        import torch
                        # Project needle to memory space for comparison
                        needle_t = torch.from_numpy(needle_emb).float().unsqueeze(0)
                        if mem.input_proj is not None:
                            needle_proj = mem.input_proj(needle_t).detach().squeeze(0).numpy()
                        else:
                            needle_proj = needle_emb.copy()
                        # Compare with spatial part of recalled hyperbolic vectors
                        sims = []
                        for v in recalled:
                            v_spatial = v[1:] if v.shape[0] > needle_proj.shape[0] else v
                            d = min(needle_proj.shape[0], v_spatial.shape[0])
                            sims.append(float(np.dot(needle_proj[:d] / (np.linalg.norm(needle_proj[:d]) + 1e-8),
                                                     v_spatial[:d] / (np.linalg.norm(v_spatial[:d]) + 1e-8))))
                        retrieved = max(sims) > 0.15
                    else:
                        retrieved = False
                elif backend_name == "tree":
                    recalled = mem.recall(query_emb, top_k=10)
                    retrieved = any(
                        needle_content[:15] in (r.get("content") or "")
                        for r in recalled
                    )
                else:
                    retrieved = False

                elapsed_ms = (time.perf_counter() - t0) * 1000

                # Memory info
                if mem is not None:
                    info = mem.info()
                    mbytes = info.get("memory_bytes", 0)
                    nodes = info.get("nodes", 0)
                    depth = info.get("max_depth", 0)
                else:
                    mbytes = nodes = depth = 0

                results.append(EvalResult(
                    backend=backend_name,
                    context_turns=num_turns,
                    needle_position_pct=pos_pct,
                    needle_retrieved=retrieved,
                    retrieval_time_ms=elapsed_ms,
                    memory_bytes=mbytes,
                    nodes=nodes,
                    depth=depth,
                ))

                del mem
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    num_turns_list: List[int] = (10, 50, 100, 500),
    needles_per_config: int = 5,
    positions: List[float] = (0.25, 0.50, 0.75),
) -> Dict[str, List[EvalResult]]:
    print("=" * 70)
    print("  Needle-in-a-Haystack Benchmark")
    print("  Comparing Tree Memory vs Flat ICM vs No Memory")
    print("=" * 70)
    print(f"  Context turns: {num_turns_list}")
    print(f"  Needles per config: {needles_per_config}")
    print(f"  Needle positions: {positions}")
    print()

    all_results = {}
    for backend_name in ("tree", "flat", "baseline"):
        print(f"  Evaluating: {backend_name} ...")
        t0 = time.perf_counter()
        results = evaluate_backend(
            backend_name,
            num_turns_list=num_turns_list,
            needles_per_config=needles_per_config,
            positions=positions,
        )
        dt = time.perf_counter() - t0
        total = len(results)
        retrieved = sum(1 for r in results if r.needle_retrieved)
        print(f"    -> {total} trials, {retrieved} retrieved, {dt:.1f}s")
        all_results[backend_name] = results

    return all_results


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(all_results: Dict[str, List[EvalResult]]):
    print()
    print("=" * 100)
    print("  RESULTS SUMMARY")
    print("=" * 100)
    print(f"  {'Backend':<12s} {'Turns':>6s} {'Pos':>5s} {'Acc':>6s} {'Lat(ms)':>8s} {'Mem(B)':>10s} {'Nodes':>6s}")
    print("  " + "-" * 55)

    for backend in ("tree", "flat", "baseline"):
        results = all_results[backend]
        turns_list = sorted(set(r.context_turns for r in results))
        for t in turns_list:
            group = [r for r in results if r.context_turns == t]
            if not group:
                continue
            acc = sum(1 for r in group if r.needle_retrieved) / len(group)
            lat = np.mean([r.retrieval_time_ms for r in group])
            mem = max(r.memory_bytes for r in group)
            nodes = max(r.nodes or 0 for r in group)
            print(f"  {backend:<12s} {t:>6d} {'all':>5s} {acc:>5.0%} {lat:>7.2f} {mem:>9,d} {nodes:>6d}")

    print()


def plot_results(all_results: Dict[str, List[EvalResult]], save_path: str = "benchmark_results.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping chart")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    backends = ["tree", "flat", "baseline"]
    colors = {"tree": "#2ecc71", "flat": "#3498db", "baseline": "#e74c3c"}
    markers = {"tree": "o", "flat": "s", "baseline": "x"}

    from collections import defaultdict
    acc = defaultdict(list)
    lat = defaultdict(list)
    mem = defaultdict(list)
    turns_list = sorted(set(r.context_turns for rr in all_results.values() for r in rr))

    for backend in backends:
        for t in turns_list:
            group = [r for r in all_results.get(backend, []) if r.context_turns == t]
            if group:
                acc[backend].append((t, sum(1 for r in group if r.needle_retrieved) / len(group)))
                lat[backend].append((t, float(np.mean([r.retrieval_time_ms for r in group]))))
                mem[backend].append((t, max(r.memory_bytes for r in group)))

    # Accuracy
    ax = axes[0]
    for b in backends:
        pts = acc[b]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=markers[b], color=colors[b],
                    label=b.upper(), linewidth=2, markersize=8)
    ax.set_xlabel("Context Turns", fontsize=12)
    ax.set_ylabel("Recall Accuracy", fontsize=12)
    ax.set_title("Needle Retrieval Accuracy", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Latency
    ax = axes[1]
    for b in ("tree", "flat"):
        pts = lat[b]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=markers[b], color=colors[b],
                    label=b.upper(), linewidth=2, markersize=8)
    ax.set_xlabel("Context Turns", fontsize=12)
    ax.set_ylabel("Retrieval Latency (ms)", fontsize=12)
    ax.set_title("Retrieval Speed", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Memory
    ax = axes[2]
    for b in ("tree", "flat"):
        pts = mem[b]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=markers[b], color=colors[b],
                    label=b.upper(), linewidth=2, markersize=8)
    ax.set_xlabel("Context Turns", fontsize=12)
    ax.set_ylabel("Memory (bytes)", fontsize=12)
    ax.set_title("Memory Footprint", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import torch
    print(f"  PyTorch {torch.__version__} | Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()

    all_results = run_benchmark(
        num_turns_list=[10, 50, 100, 500],
        needles_per_config=5,
        positions=[0.25, 0.50, 0.75],
    )

    print_summary(all_results)
    plot_results(all_results, "benchmark_results.png")

    # Final numbers
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    for backend in ("tree", "flat", "baseline"):
        results = all_results[backend]
        total = len(results)
        retrieved = sum(1 for r in results if r.needle_retrieved)
        avg_lat = float(np.mean([r.retrieval_time_ms for r in results]))
        max_mem = max(r.memory_bytes for r in results)
        print(f"  {backend:<10s}: {retrieved}/{total} recalled "
              f"({retrieved/total*100:.0f}%), "
              f"avg {avg_lat:.2f}ms, max {max_mem:,}B")

    print()
    return all_results


if __name__ == "__main__":
    main()
