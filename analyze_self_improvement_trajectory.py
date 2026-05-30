"""
Detailed Analysis of a Self-Improvement Trajectory

Loads a trajectory JSON produced by run_self_improvement_loop.py and produces
a rigorous, quantitative breakdown of what happened and why.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics

def analyze_trajectory(json_path: str = "self_improvement_trajectory.json"):
    with open(json_path) as f:
        data = json.load(f)

    print("=" * 75)
    print("DETAILED SELF-IMPROVEMENT TRAJECTORY ANALYSIS")
    print("=" * 75)
    print()

    trajectory = data["trajectory"]
    initial = data["initial_utility"]
    final = data["final_utility"]
    total_gain = data["total_gain"]

    print(f"Initial Utility: {initial:.2f}")
    print(f"Final Utility:   {final:.2f}")
    print(f"Total Gain:      +{total_gain:.2f}")
    print(f"Number of iterations: {len(trajectory)}")
    print()

    # === Basic Statistics ===
    gains = [step["utility_gain"] for step in trajectory]
    accepted_gains = [g for g in gains if g > 0.1]

    print("--- Gain Statistics ---")
    print(f"Mean gain per iteration:     {statistics.mean(gains):.3f}")
    print(f"Median gain per iteration:   {statistics.median(gains):.3f}")
    if accepted_gains:
        print(f"Mean gain on accepted steps: {statistics.mean(accepted_gains):.3f}")
    print(f"Number of steps with positive gain: {sum(1 for g in gains if g > 0.1)} / {len(gains)}")
    print()

    # === Config Evolution ===
    print("--- Hyperbolic Loss Weight Evolution (Primary Lever) ---")
    hyp_weights = []
    for step in trajectory:
        w = step["after_config"]["hyperbolic_loss_weight"]
        hyp_weights.append(w)
        print(f"  After iter {step['iteration']}: {w:.5f}")

    if len(hyp_weights) > 1:
        total_increase = hyp_weights[-1] - hyp_weights[0]
        print(f"\n  Total increase in hyp_loss_weight: {total_increase:.5f} ({total_increase/hyp_weights[0]*100:.1f}%)")
    print()

    # === Effectiveness of Different Change Types ===
    print("--- Change Type Effectiveness ---")
    change_types = defaultdict(list)

    for step in trajectory:
        for change in step.get("accepted_changes", []):
            rationale = change.get("rationale", "").lower()
            actual = change.get("actual_gain", 0.0)

            if "hyperbolic" in rationale or "geometric loss" in rationale:
                change_types["Increase geometric loss weight"].append(actual)
            elif "tile" in rationale:
                change_types["Increase tile size"].append(actual)
            elif "entropy" in rationale or "router" in rationale:
                change_types["Increase router entropy weight"].append(actual)
            elif "fusion" in rationale:
                change_types["Enable/strengthen fusion"].append(actual)
            else:
                change_types["Other"].append(actual)

    for ctype, gains_list in sorted(change_types.items(), key=lambda x: -sum(x[1])):
        if gains_list:
            avg = statistics.mean(gains_list)
            print(f"  {ctype:40s} | times accepted: {len(gains_list):2d} | avg actual gain: {avg:+.2f}")

    print()

    # === Diminishing Returns Analysis ===
    print("--- Diminishing Returns / Plateau Detection ---")
    if len(gains) >= 3:
        early_gain = sum(gains[:2])
        late_gain = sum(gains[-2:])
        print(f"  Gain in first 2 iterations:  {early_gain:+.2f}")
        print(f"  Gain in last 2 iterations:   {late_gain:+.2f}")

        if late_gain < early_gain * 0.4:
            print("  → Clear diminishing returns observed. The system correctly slowed down.")
        elif late_gain > early_gain * 0.8:
            print("  → Still strong gains in later iterations. More headroom may exist.")
        else:
            print("  → Moderate slowdown in improvement rate.")

    print()

    # === Recommendations based on data ===
    print("--- Data-Driven Recommendations ---")
    if total_gain < 1.0:
        print("  • Very small total improvement. Consider expanding the set of modifiable knobs significantly.")
    if "Increase geometric loss weight" in change_types and len(change_types["Increase geometric loss weight"]) >= 3:
        print("  • The system relied heavily on increasing hyperbolic loss weight. This suggests the original default was too conservative.")
    if any("tile" in str(c).lower() for c in change_types):
        print("  • Tile size changes were attempted — monitor whether larger tiles actually help long-range recall in bigger experiments.")
    print("  • Next step: Allow the reasoner to also propose changes to router_entropy_weight and attention_every_n more aggressively.")

    print()
    print("=" * 75)
    print("Analysis complete. Raw data is in the trajectory JSON.")
    print("=" * 75)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "self_improvement_trajectory.json"
    analyze_trajectory(path)