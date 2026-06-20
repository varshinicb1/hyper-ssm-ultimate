"""
Proper Fused vs Non-Fused Training Comparison Script

This script runs two parallel training experiments:
- Baseline: Standard Hyper-SSM without GeometryAwareParallelFusion
- Fused: Same architecture but with --use_geometry_fusion enabled

It uses the production trainer and logs key metrics so you can compare:
- Training speed / loss curves
- Recall / long-range performance
- Manifold stability
- Effect of fusion on final quality

Usage (CPU demo):
    python scripts/compare_fused_vs_baseline_training.py --steps 800 --dim 128

For real GPU runs, launch two instances with the production flags.
"""

import subprocess
import sys
from pathlib import Path
import time
import json

ROOT = Path(__file__).resolve().parents[1]

def run_training(mode: str, steps: int, dim: int, tag: str):
    cmd = [
        sys.executable,
        str(ROOT / "training" / "train_hybrid_ultimate.py"),
        "--use_tiled",
        "--hidden_size", str(dim),
        "--num_layers", "8",
        "--max_steps", str(steps),
        "--batch", "4",
        "--seq_len", "512",
        "--log_interval", "50",
        "--run_name", f"{tag}_{mode}",
    ]

    if mode == "fused":
        cmd.extend([
            "--use_geometry_fusion",
            "--fusion_mode", "tangent_gated",
        ])

    print(f"\n=== Starting {mode.upper()} run ({tag}) ===")
    print("Command:", " ".join(cmd))

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    duration = time.time() - start

    # Save logs
    log_dir = ROOT / "logs" / f"comparison_{tag}"
    log_dir.mkdir(parents=True, exist_ok=True)

    (log_dir / f"{mode}_stdout.log").write_text(result.stdout)
    (log_dir / f"{mode}_stderr.log").write_text(result.stderr)

    print(f"{mode.upper()} finished in {duration/60:.1f} min")
    print(f"Logs saved to {log_dir}")

    return {
        "mode": mode,
        "duration_minutes": round(duration / 60, 2),
        "returncode": result.returncode,
        "log_dir": str(log_dir)
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--tag", type=str, default="demo")
    args = parser.parse_args()

    print("=== FUSED vs BASELINE COMPARISON ===")
    print(f"Steps: {args.steps} | Hidden: {args.dim} | Tag: {args.tag}\n")

    results = []
    results.append(run_training("baseline", args.steps, args.dim, args.tag))
    results.append(run_training("fused", args.steps, args.dim, args.tag))

    summary_path = ROOT / "logs" / f"comparison_{args.tag}" / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))

    print("\n=== COMPARISON COMPLETE ===")
    print("Summary saved to:", summary_path)
    print("Check the individual log files for fused vs non-fused metrics.")
    print("Look for 'geometry_fusion_active' in the fused run logs.")

if __name__ == "__main__":
    main()
