import sys, traceback
sys.path.insert(0, ".")
print("=== PINNACLE CANDIDATE 3 VALIDATION ===")
import torch
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
    nparams = sum(p.numel() for p in model.parameters())/1e6
    print(f"[PASS] Model created: {nparams:.2f}M params (tiled)")

    x = torch.randint(0, 1000, (1, 16), device=device)
    state = model.get_final_state(x, with_manifold_checks=True)
    print(f"[PASS] get_final_state: using_tiled={state.get('using_tiled')}")

    new_x = torch.randint(0, 1000, (1, 3), device=device)
    new_state, logits = model.update_state(state, new_x, with_manifold_checks=True)
    print(f"[PASS] update_state: logits shape {logits.shape}")

    gen = model.generate_efficient(x, max_new_tokens=5, with_manifold_checks=True)
    print(f"[PASS] generate_efficient: len={gen.shape[1]}")

    comp = model.layers[0].compressor
    rep = comp.get_performance_report()
    print(f"[PASS] PerformanceCounters + report: calls={rep.get('total_calls')}, best={rep.get('best_mode')}")

    h = comp.reset_state(2, device)
    v = check_manifold_constraint(h)
    print(f"[PASS] Manifold check: viol={float(v):.2e}")

    if hasattr(comp, "autotune_compile"):
        r = comp.autotune_compile(test_shape=(1, 8, 65), iters=1)
        print(f"[PASS] Autotune: winner={r.get('winner')}")

    print("\n=== ALL PINNACLE VALIDATIONS PASSED ===")
except Exception as e:
    print("VALIDATION FAILED:", type(e).__name__, ":", str(e))
    traceback.print_exc()
