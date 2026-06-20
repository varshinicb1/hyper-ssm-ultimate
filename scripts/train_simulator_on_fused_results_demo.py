"""
Demonstration: Training the Experimental Simulator's surrogate on fused execution results.

This shows the continuous learning loop in action:
1. Start with heuristic-only simulator.
2. Run several "executions" (simulated).
3. Feed results back → train the neural surrogate.
4. Show that predictions improve after training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project-aether" / "src"))

from aether.simulation.experimental_simulator import ExperimentalSimulator
from aether.schemas.core import ExperimentalPlan, SynthesisProtocol, Material, SynthesisStep, HydrothermalConditions
import random

def create_sample_plan(temp=180, time=12, ph=7.0, dopant=False):
    return ExperimentalPlan(
        hypothesis_id="demo-plan",
        target_material_formula="TiO2",
        recommended_protocol=SynthesisProtocol(
            target_material=Material(name="TiO2", formula="TiO2"),
            steps=[SynthesisStep(
                step_number=1,
                step_type="hydrothermal",
                conditions=HydrothermalConditions(temperature_c=temp, duration_hours=time, pH=ph),
                dopants=[{"name": "N-dopant"}] if dopant else []
            )]
        )
    )

def main():
    print("=== Training the Experimental Simulator Surrogate ===\n")

    sim = ExperimentalSimulator()  # surrogate training works independently of fusion for now

    # Generate fake historical execution results (as if from robotic runs)
    training_examples = []
    for _ in range(60):
        temp = random.randint(150, 230)
        time = random.randint(6, 40)
        ph = round(random.uniform(1.8, 11.0), 1)
        dopant = random.random() > 0.5

        plan = create_sample_plan(temp, time, ph, dopant)

        # Simulate "real" outcome with some noise
        true_success = 0.7
        if temp > 210: true_success -= 0.25
        if time < 8: true_success -= 0.2
        if dopant: true_success += 0.1
        actual = max(0.0, min(1.0, true_success + random.gauss(0, 0.15)))

        training_examples.append((plan, actual))

    print(f"Collected {len(training_examples)} historical execution results.\n")

    # Before training
    print("Before training surrogate:")
    test_plan = create_sample_plan(195, 18, 3.2, dopant=True)
    before = sim.simulate(test_plan)
    print(f"  Predicted success: {before.predicted_success_probability:.3f} | Outcome: {before.predicted_outcome}")

    # Train on the results
    print("\nTraining surrogate on execution results...")
    for plan, actual in training_examples:
        sim.train_on_result(plan, actual)

    print(f"Surrogate now trained on {len(sim.training_data)} examples.\n")

    # After training
    print("After training surrogate:")
    after = sim.simulate(test_plan)
    print(f"  Predicted success: {after.predicted_success_probability:.3f} | Outcome: {after.predicted_outcome}")
    print(f"  Simulator notes: {after.notes}")

    print("\n=== The simulator is now learning from fused robotic feedback ===")

if __name__ == "__main__":
    main()
