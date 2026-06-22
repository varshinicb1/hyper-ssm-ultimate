"""Generate figures for the ICM paper.

Usage:
  python paper/generate_figures.py

Requires matplotlib. Saves figures to paper/figures/.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})

def fig_niah_accuracy():
    """NIAH recall accuracy: Tree vs Flat vs Baseline."""
    turns = [10, 50, 100, 500]
    tree = [100, 100, 100, 100]
    flat = [40, 27, 20, 27]
    base = [0, 0, 0, 0]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(turns, tree, "o-", color="#3b82f6", linewidth=2, label="Tree (O(log N))")
    ax.plot(turns, flat, "s-", color="#f59e0b", linewidth=2, label="Flat (O(1))")
    ax.plot(turns, base, "d-", color="#ef4444", linewidth=2, label="Baseline")
    ax.set_xscale("log")
    ax.set_xlabel("Context Turns")
    ax.set_ylabel("Recall Accuracy (%)")
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_title("NIAH: Recall Accuracy vs Context Length")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "niah_accuracy.pdf"))
    fig.savefig(os.path.join(OUT, "niah_accuracy.png"))
    plt.close(fig)
    print(f"  -> {OUT}/niah_accuracy.pdf, .png")

def fig_memory_scaling():
    """Memory footprint comparison: KV-cache vs Flat vs Tree."""
    turns = np.array([10, 100, 1000, 10000])
    kv_gb = np.array([1.2, 12, 120, 1200])  # GB for 7B model
    flat_mb = np.full_like(turns, 260 / 1024 / 1024)  # 260B -> MB
    tree_mb = np.array([31, 301, 2800, 28000]) / 1024  # KB -> MB

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.loglog(turns, kv_gb * 1024, "o-", color="#ef4444", linewidth=2, label="KV-Cache (7B)")
    ax.loglog(turns, flat_mb, "s-", color="#22c55e", linewidth=2, label="Flat (O(1))")
    ax.loglog(turns, tree_mb, "d-", color="#3b82f6", linewidth=2, label="Tree (O(log N))")
    ax.set_xlabel("Context Turns")
    ax.set_ylabel("Memory (MB)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title("Memory Scaling: ICM vs KV-Cache")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "memory_scaling.pdf"))
    fig.savefig(os.path.join(OUT, "memory_scaling.png"))
    plt.close(fig)
    print(f"  -> {OUT}/memory_scaling.pdf, .png")

def fig_insertion_speed():
    """Insertion time with and without lazy updates."""
    facts = np.array([100, 500, 1000])
    no_lazy = np.array([0.4, 5.2, 14.1])
    lazy = np.array([0.1, 1.1, 2.0])

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(facts, no_lazy, "o-", color="#ef4444", linewidth=2, label="No Lazy Update")
    ax.plot(facts, lazy, "s-", color="#3b82f6", linewidth=2, label="Lazy Update")
    ax.set_xlabel("Facts Stored")
    ax.set_ylabel("Insertion Time (s)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_title("Tree Insertion: Lazy vs No Lazy")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "insertion_speed.pdf"))
    fig.savefig(os.path.join(OUT, "insertion_speed.png"))
    plt.close(fig)
    print(f"  -> {OUT}/insertion_speed.pdf, .png")

if __name__ == "__main__":
    print("Generating figures...")
    fig_niah_accuracy()
    fig_memory_scaling()
    fig_insertion_speed()
    print("Done.")
