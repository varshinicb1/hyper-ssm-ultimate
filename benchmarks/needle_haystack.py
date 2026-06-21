"""
Needle-in-a-Haystack benchmark: Hyperbolic Memory Tree vs Flat ICM vs No Memory.

Measures recall accuracy, retrieval latency, and memory usage at
increasing context lengths (10 to 1000 turns).
"""

import time
import gc
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

FILLER_TEMPLATES = [
    "The weather today is {adj} with a high of {temp}°C.",
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
    "temp": list(range(15, 38)),
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
              "meditate", "solve a Rubik's cube"],
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


def _fill(template: str) -> str:
    import random
    while "{placeholder}" in template:
        template = template.replace("{placeholder}", random.choice(FILLER_VOCAB.get("place", ["nowhere"])), 1)
    for key, values in FILLER_VOCAB.items():
        placeholder = f"{{{key}}}"
        if placeholder in template:
            template = template.replace(placeholder, str(random.choice(values)), 1)
    return template


def generate_filler_turn(turn_idx: int, rng: np.random.Generator) -> str:
    template = FILLER_TEMPLATES[turn_idx % len(FILLER_TEMPLATES)]
    return _fill(template)


NEEDLE_TEMPLATES = [
    "The secret code is {code}. Remember it: {code}.",
    "The password for the vault is {word}-{num}.",
    "The hidden location is at coordinates {lat}, {lon}.",
    "The access key is {key}. Keep it safe.",
    "The backup date is {month} {day}, {year}.",
]


def generate_needle(needle_id: int, rng: np.random.Generator) -> Tuple[str, str]:
    template = NEEDLE_TEMPLATES[needle_id % len(NEEDLE_TEMPLATES)]
    code = f"{rng.integers(100, 999)}X{rng.integers(10, 99)}"
    word = rng.choice(["Zephyr", "Nebula", "Aether", "Quantum", "Phoenix",
                        "Cascade", "Horizon", "Tempest", "Aurora", "Onyx"])
    num = rng.integers(1000, 9999)
    lat = f"{rng.integers(-90, 90)}.{rng.integers(0, 999):03d}"
    lon = f"{rng.integers(-180, 180)}.{rng.integers(0, 999):03d}"
    key = f"{rng.integers(100000, 999999):06d}"
    month = rng.choice(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    day = rng.integers(1, 29)
    year = rng.integers(2020, 2030)
    content = _fill(template.format(
        code=code, word=word, num=num, lat=lat, lon=lon,
        key=key, month=month, day=day, year=year,
    ))
    # Generate a query that should retrieve this needle
    query = f"What is the secret code/password/location/key/date?"
    return query, content


# ---------------------------------------------------------------------------
# Backend wrappers
# ---------------------------------------------------------------------------

def make_flat(embed_dim: int = 384, state_dim: int = 64, num_scales: int = 4):
    from hyper_ssm.conversation_memory import InfiniteContextMemory
    return InfiniteContextMemory(
        embedding_dim=embed_dim,
        state_dim=state_dim,
        num_scales=num_scales,
        device=torch.device("cpu"),
    )

def make_tree(embed_dim: int = 384, state_dim: int = 64, max_depth: int = 10):
    from hyper_ssm.memory_tree import HyperbolicMemoryTree
    return HyperbolicMemoryTree(
        embed_dim=embed_dim,
        state_dim=state_dim,
        max_depth=max_depth,
    )

def make_baseline():
    return None


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _embedder = None
    return _embedder


def embed(text: str) -> np.ndarray:
    emb = get_embedder()
    if emb is not None:
        return emb.encode(text, normalize_embeddings=True).astype(np.float32)
    return np.random.randn(384).astype(np.float32)


# ---------------------------------------------------------------------------
# Single evaluation
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


def evaluate_backend(
    backend_name: str,
    backend,
    embed_dim: int = 384,
    num_turns_list: List[int] = (10, 50, 100, 500, 1000),
    needles_per_config: int = 3,
    positions: List[float] = (0.25, 0.50, 0.75),
) -> List[EvalResult]:
    import random as py_random

    results = []

    for num_turns in num_turns_list:
        for ni in range(needles_per_config):
            for pos_pct in positions:
                seed = hash(f"{backend_name}_{num_turns}_{ni}_{pos_pct}") & 0xFFFFFFFF
                rng = np.random.default_rng(seed)
                py_random.seed(seed)

                # Recreate backend for each trial
                if backend_name == "flat":
                    mem = make_flat(embed_dim=embed_dim)
                elif backend_name == "tree":
                    mem = make_tree(embed_dim=embed_dim)
                else:
                    mem = None

                # Generate needle
                _, needle_content = generate_needle(ni, rng)
                needle_emb = embed(needle_content)
                query = f"Tell me the secret code?"
                query_emb = embed(query)

                # Insert filler turns around the needle
                needle_turn = int(num_turns * pos_pct)
                for i in range(num_turns):
                    if i == needle_turn:
                        if backend_name == "tree" and mem is not None:
                            mem.remember(needle_emb, needle_content)
                        elif mem is not None:
                            mem.remember(needle_emb)
                    else:
                        filler = generate_filler_turn(i, rng)
                        filler_emb = embed(filler)
                        if backend_name == "tree" and mem is not None:
                            mem.remember(filler_emb, filler)
                        elif mem is not None:
                            mem.remember(filler_emb)

                # Time the retrieval
                retrieved = False
                t0 = time.perf_counter()

                if backend_name == "baseline" or mem is None:
                    retrieved = False
                elif backend_name == "flat":
                    recalled = mem.recall_all_scales(query_emb)
                    # Check if needle embedding is closest to any recalled vector
                    if len(recalled) > 0:
                        sims = [float(np.dot(needle_emb, v)) for v in recalled if v is not None and len(v) > 0]
                        retrieved = any(s > 0.5 for s in sims)
                elif backend_name == "tree":
                    recalled = mem.recall(query_emb, top_k=10)
                    retrieved = any(needle_content[:20] in (r.get("content") or "") for r in recalled)

                elapsed_ms = (time.perf_counter() - t0) * 1000

                # Memory info
                if mem is not None:
                    info = mem.info()
                    mbytes = info.get("memory_bytes", 0)
                    nodes = info.get("nodes")
                    depth = info.get("max_depth")
                else:
                    mbytes = 0
                    nodes = 0
                    depth = 0

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

                # Cleanup
                del mem
                gc.collect()
                import torch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    num_turns_list: List[int] = (10, 50, 100, 500, 1000),
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
        print(f"  Evaluating backend: {backend_name} ...")
        t0 = time.perf_counter()
        results = evaluate_backend(
            backend_name, None,
            num_turns_list=num_turns_list,
            needles_per_config=needles_per_config,
            positions=positions,
        )
        dt = time.perf_counter() - t0
        total = len(results)
        retrieved = sum(1 for r in results if r.needle_retrieved)
        print(f"    Done: {total} trials, {retrieved} retrieved, {dt:.1f}s")
        all_results[backend_name] = results

    return all_results


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

def print_summary(all_results: Dict[str, List[EvalResult]]):
    import pandas as pd

    print()
    print("=" * 100)
    print("  RESULTS SUMMARY")
    print("=" * 100)

    rows = []
    for backend, results in all_results.items():
        for r in results:
            rows.append({
                "Backend": backend.upper(),
                "Turns": r.context_turns,
                "Position": f"{r.needle_position_pct:.0%}",
                "Retrieved": r.needle_retrieved,
                "Latency (ms)": f"{r.retrieval_time_ms:.2f}",
                "Memory (B)": r.memory_bytes,
                "Nodes": r.nodes or 0,
            })

    df = pd.DataFrame(rows)
    summary = df.groupby(["Backend", "Turns"]).agg(
        Accuracy=("Retrieved", "mean"),
        Latency_ms=("Latency (ms)", lambda x: f"{x.astype(float).mean():.2f}"),
        Memory_B=("Memory (B)", "max"),
    ).reset_index()

    print(summary.to_string(index=False))
    print()
    return df


def plot_results(all_results: Dict[str, List[EvalResult]], save_path: str = "benchmark_results.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    backends = list(all_results.keys())
    colors = {"tree": "#2ecc71", "flat": "#3498db", "baseline": "#e74c3c"}
    markers = {"tree": "o", "flat": "s", "baseline": "x"}

    # Aggregate by backend + turns
    from collections import defaultdict
    acc = defaultdict(list)
    lat = defaultdict(list)
    mem = defaultdict(list)
    turns_list = sorted(set(r.context_turns for rr in all_results.values() for r in rr))

    for backend in backends:
        for t in turns_list:
            group = [r for r in all_results[backend] if r.context_turns == t]
            if group:
                acc[backend].append((t, sum(1 for r in group if r.needle_retrieved) / len(group)))
                lat[backend].append((t, np.mean([r.retrieval_time_ms for r in group])))
                mem[backend].append((t, max(r.memory_bytes for r in group)))

    # Panel 1: Accuracy vs Context Turns
    ax = axes[0]
    for backend in backends:
        points = acc[backend]
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker=markers[backend], color=colors[backend],
                    label=backend.upper(), linewidth=2, markersize=8)
    ax.set_xlabel("Context Turns", fontsize=12)
    ax.set_ylabel("Recall Accuracy", fontsize=12)
    ax.set_title("Needle Retrieval Accuracy", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 2: Latency vs Context Turns
    ax = axes[1]
    for backend in ("tree", "flat"):
        points = lat[backend]
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker=markers[backend], color=colors[backend],
                    label=backend.upper(), linewidth=2, markersize=8)
    ax.set_xlabel("Context Turns", fontsize=12)
    ax.set_ylabel("Retrieval Latency (ms)", fontsize=12)
    ax.set_title("Retrieval Speed", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 3: Memory Usage vs Context Turns
    ax = axes[2]
    for backend in ("tree", "flat"):
        points = mem[backend]
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker=markers[backend], color=colors[backend],
                    label=backend.upper(), linewidth=2, markersize=8)
    ax.set_xlabel("Context Turns", fontsize=12)
    ax.set_ylabel("Memory (bytes)", fontsize=12)
    ax.set_title("Memory Footprint", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()

    all_results = run_benchmark(
        num_turns_list=[10, 50, 100, 500],
        needles_per_config=5,
        positions=[0.25, 0.50, 0.75],
    )

    df = print_summary(all_results)
    plot_results(all_results, "benchmark_results.png")

    # Print final verdict
    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    for backend in ("tree", "flat", "baseline"):
        results = all_results[backend]
        total = len(results)
        retrieved = sum(1 for r in results if r.needle_retrieved)
        avg_lat = np.mean([r.retrieval_time_ms for r in results])
        max_mem = max(r.memory_bytes for r in results)
        print(f"  {backend.upper():10s}: {retrieved}/{total} recalled "
              f"({retrieved/total*100:.0f}%), "
              f"avg {avg_lat:.2f}ms, max {max_mem:,}B")

    print()
    return all_results


if __name__ == "__main__":
    results = main()
