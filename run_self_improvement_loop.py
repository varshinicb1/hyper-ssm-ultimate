"""
Run a longer self-improvement loop for Hyper-SSM / Project Aether.

This script:
- Instantiates the SelfImprovementEngine (wired to real benchmark)
- Runs multiple iterations
- Records the full trajectory: utility before/after, accepted changes, config evolution
- Saves results to JSON for analysis

This demonstrates mathematical, goal-directed self-modification in action.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add paths
ROOT = Path(__file__).resolve()
sys.path.insert(0, str(ROOT / "project-aether" / "src"))
sys.path.insert(0, str(ROOT))

from aether.self_improvement.engine import SelfImprovementEngine


class DummyMemory:
    """Placeholder - in a full system this would be the real ScientificMemoryEngine"""
    pass


class DummyReasoner:
    """Placeholder - in future this would be FullHyperSSMReasoner generating proposals"""
    pass


class DummySimulator:
    """Placeholder - in future this would be ExperimentalSimulator"""
    pass


def run_long_self_improvement_loop(
    num_iterations: int = 6,
    proposals_per_step: int = 2,
    output_file: str = "self_improvement_trajectory.json"
):
    print("=" * 70)
    print("LONG SELF-IMPROVEMENT LOOP - Hyper-SSM / Project Aether")
    print("=" * 70)
    print(f"Goal: Maximize hierarchical recall utility (recall@32)")
    print(f"Iterations: {num_iterations}")
    print(f"Proposals per step: {proposals_per_step}")
    print()

    engine = SelfImprovementEngine(
        memory_engine=DummyMemory(),
        reasoner=DummyReasoner(),
        simulator=DummySimulator(),
        goal_metric="hierarchical_recall_at_32"
    )

    trajectory = []
    start_time = datetime.now()

    initial_utility = engine.get_current_utility()
    print(f"Initial Utility: {initial_utility:.2f}")
    print()

    for iteration in range(1, num_iterations + 1):
        print(f"{'='*60}")
        print(f"ITERATION {iteration}/{num_iterations}")
        print(f"{'='*60}")

        before_utility = engine.get_current_utility()
        before_config = {
            "hyperbolic_loss_weight": engine.current_config.hyperbolic_loss_weight,
            "fusion_mode": engine.current_config.fusion_mode,
            "tile_size": engine.current_config.tile_size,
        }

        print(f"Before utility: {before_utility:.2f}")
        print(f"Current config: {before_config}")

        # Run one self-improvement step
        engine.run_self_improvement_step(max_proposals=proposals_per_step)

        after_utility = engine.get_current_utility()
        after_config = {
            "hyperbolic_loss_weight": engine.current_config.hyperbolic_loss_weight,
            "fusion_mode": engine.current_config.fusion_mode,
            "tile_size": engine.current_config.tile_size,
        }

        gain = after_utility - before_utility

        # Record this iteration
        iter_record = {
            "iteration": iteration,
            "before_utility": round(before_utility, 2),
            "after_utility": round(after_utility, 2),
            "utility_gain": round(gain, 2),
            "before_config": before_config,
            "after_config": after_config,
            "accepted_changes": []
        }

        # Analyze what happened in this step from history
        if engine.improvement_history:
            recent = engine.improvement_history[-2:] if len(engine.improvement_history) > 1 else engine.improvement_history[-1:]
            for result in recent:
                if result.accepted:
                    iter_record["accepted_changes"].append({
                        "rationale": result.proposal.rationale,
                        "delta": result.proposal.config_delta,
                        "predicted_gain": result.proposal.expected_utility_gain,
                        "actual_gain": round(result.actual_gain, 2)
                    })

        trajectory.append(iter_record)

        print(f"After utility:  {after_utility:.2f}")
        print(f"Gain this step: {gain:+.2f}")
        print(f"New config:     {after_config}")

        if iter_record["accepted_changes"]:
            print("Accepted improvements this step:")
            for ch in iter_record["accepted_changes"]:
                print(f"  - {ch['rationale']} (actual gain: {ch['actual_gain']:+.2f})")
        else:
            print("No improvements accepted this iteration.")

        print()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Final summary
    final_utility = engine.get_current_utility()
    total_gain = final_utility - initial_utility

    summary = {
        "experiment": "self_improvement_loop",
        "timestamp": start_time.isoformat(),
        "duration_seconds": round(duration, 1),
        "num_iterations": num_iterations,
        "initial_utility": round(initial_utility, 2),
        "final_utility": round(final_utility, 2),
        "total_gain": round(total_gain, 2),
        "final_config": {
            "hyperbolic_loss_weight": engine.current_config.hyperbolic_loss_weight,
            "fusion_mode": engine.current_config.fusion_mode,
            "tile_size": engine.current_config.tile_size,
        },
        "trajectory": trajectory,
        "notes": [
            "Utilities come from real runs of evaluate_system_config (hierarchical recall benchmark).",
            "Each utility evaluation trains a small model for ~120 steps.",
            "This is a real, measurable self-improvement trajectory."
        ]
    }

    # Save
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("SELF-IMPROVEMENT LOOP COMPLETE")
    print("=" * 70)
    print(f"Initial Utility: {initial_utility:.2f}")
    print(f"Final Utility:   {final_utility:.2f}")
    print(f"Total Gain:      {total_gain:+.2f}")
    print(f"Duration:        {duration:.1f} seconds")
    print(f"Trajectory saved to: {output_file}")
    print()

    # Pretty print trajectory
    print("TRAJECTORY:")
    print("-" * 70)
    for step in trajectory:
        print(f"Iter {step['iteration']:2d}: {step['before_utility']:5.2f} → {step['after_utility']:5.2f} "
              f"(gain {step['utility_gain']:+.2f})  |  hyp_weight={step['after_config']['hyperbolic_loss_weight']:.4f}")

    return summary


if __name__ == "__main__":
    run_long_self_improvement_loop(
        num_iterations=5,
        proposals_per_step=2,
        output_file="self_improvement_trajectory.json"
    )