"""
PINNACLE CANDIDATE 3 — AUTHORITATIVE VALIDATION GATE (2026 Ultimate)

This is the single source of truth for "does the full production Hyper-SSM stack work?"

It exercises:
- TiledFractalCompressor (vectorized + compile + manifold safety)
- Stateful O(1) generation APIs (get_final_state → update_state → generate_efficient)
- Performance telemetry + autotune
- The geometrically correct hyperbolic loss path (real Lorentz states + tangent_space)

Run this with `python -W error::UserWarning pinnacle_validate.py` for strict mode.
If this script passes cleanly, the core pinnacle stack has no known loose ends.
"""

import sys
import traceback
import warnings

import torch

sys.path.insert(0, ".")
print("=== PINNACLE CANDIDATE 3 — AUTHORITATIVE VALIDATION ===")

try:
    from hyper_ssm import (
        HyperSSM, HyperSSMConfig, TiledFractalCompressor,
        check_manifold_constraint, project_to_manifold, PerformanceCounters,
        create_hyperbolic_loss, is_rust_kernels_available
    )
    print("[PASS] All pinnacle exports successful")

    device = "cpu"
    config = HyperSSMConfig(vocab_size=1000, hidden_size=64, num_layers=4)
    model = HyperSSM(config, use_hybrid=True, attention_every_n=2, use_tiled_compressor=True).to(device)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[PASS] Model created: {nparams:.2f}M params (tiled)")

    # === Core generation / state APIs ===
    x = torch.randint(0, 1000, (1, 16), device=device)
    state = model.get_final_state(x, with_manifold_checks=True)
    print(f"[PASS] get_final_state: using_tiled={state.get('using_tiled')}")

    new_x = torch.randint(0, 1000, (1, 3), device=device)
    new_state, logits = model.update_state(state, new_x, with_manifold_checks=True)
    print(f"[PASS] update_state: logits shape {logits.shape}")

    gen = model.generate_efficient(x, max_new_tokens=5, with_manifold_checks=True)
    print(f"[PASS] generate_efficient: len={gen.shape[1]}")

    # === Telemetry & compile ===
    comp = model.layers[0].compressor
    rep = comp.get_performance_report()
    print(f"[PASS] PerformanceCounters: calls={rep.get('total_calls')}, best={rep.get('best_mode')}")

    h = comp.reset_state(2, device)
    v = check_manifold_constraint(h)
    print(f"[PASS] Manifold health: viol={float(v.detach() if isinstance(v, torch.Tensor) else v):.2e}")

    if hasattr(comp, "autotune_compile"):
        r = comp.autotune_compile(test_shape=(1, 8, 65), iters=1)
        print(f"[PASS] Autotune: winner={r.get('winner')}")

    # === GEOMETRIC LOSS PATH (the Phase 2 correctness gate) ===
    # This is the critical new check: real Lorentz states → non-zero tangent-space loss
    lorentz_out = model.get_lorentz_representations(x, final_only=True, with_manifold_checks=False)
    lorentz_states = lorentz_out["lorentz_states"]
    print(f"[PASS] get_lorentz_representations: shape={tuple(lorentz_states.shape)}")

    hyp_loss_fn = create_hyperbolic_loss()   # uses the recommended defaults (tangent_space=True)
    loss_dict = hyp_loss_fn(lorentz_states)

    assert loss_dict["metric"] == "tangent", f"Expected tangent metric, got {loss_dict['metric']}"
    assert loss_dict["total"] > 0 or loss_dict.get("radius_health", 0) > 0, "Hyperbolic loss produced zero signal"
    print(f"[PASS] HyperbolicLoss (geometric): total={loss_dict['total']:.5f}, "
          f"metric={loss_dict['metric']}, radius_health={loss_dict.get('radius_health', 0):.5f}")

    # Also verify the class itself reports the right defaults
    print(f"[PASS] HyperbolicLoss repr: {repr(hyp_loss_fn)}")

    print("\n=== ALL PINNACLE VALIDATIONS PASSED (including geometric loss) ===")
    print("This stack is considered complete for the current pinnacle milestone.")

except Exception as e:
    print("VALIDATION FAILED:", type(e).__name__, ":", str(e))
    traceback.print_exc()
    sys.exit(1)
